#include "stlq/linkage/linkage_streaming_builders.h"

#include <array>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <omp.h>

#include "stlq/linkage/linkage_build_profile.h"
#include "stlq/linkage/linkage_finalize.h"
#include "stlq/linkage/linkage_streaming_support.h"
#include "stlq/linkage/linkage_summary.h"
#include "stlq/core/blas.h"
#include "stlq/core/kernel_provider_cpu.h"
#include "stlq/core/lapack.h"
#include "stlq/coeff/coeff_cluster_codec.h"
#include "stlq/io/base_list_store.h"
#include "stlq/io/bvecs_reader.h"
#include "stlq/io/dataset_reader_factory.h"
#include "stlq/io/linkage_coeff_codec_store.h"
#include "stlq/io/linkage_list_store.h"
#include "stlq/io/fbin_reader.h"
#include "stlq/io/fvecs_reader.h"
#include "stlq/io/ivf_lists.h"
#include "stlq/knn/hnsw_cluster_knn.h"
#include "stlq/common/logger.h"
#include "stlq/quantizer/encoder.h"
#include "stlq/quantizer/linear_algebra.h"
#include "stlq/succinct/parent_louds.h"
#include "stlq/linkage/virtual_augment.h"
#include "stlq/common/timer.h"

#include "stlq/cuda/cuda_stream_kernels_pool.h"
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
#include "stlq/linkage/eval_candidates_cuda.h"
#include "stlq/linkage/linkage_encode_cuda.h"
#endif

namespace stlq
{
    namespace
    {
#include "linkage_builder_common.inc"
#include "linkage_builder_two_multicenter.inc"

        using linkage::AsyncClusterPrefetcher;
        using linkage::ClusterCheckpoint;
        using linkage::ClusterCkptStatsV1;
        using linkage::ClusterWorkBuf;
        using linkage::PrefetchedClusterSpans;
        using linkage::BuildPrecompInitLinkageFixedRootTinyCpu;
        using linkage::BuildPrecompMetaOnly;
        using linkage::ReaderCanReadF32;
        using linkage::ReaderCanReadU8;
        using linkage::SolveSymPosdefCholeskyRetry;

#include "linkage_cluster_builders.inc"

    } // namespace

    bool BuildLinkageTwoCodebookVirtualStreamingByCluster(const Config& cfg,
                                                          const TrainResult& train,
                                                          const VirtualUmapReencodeEncodeConfig& umap_reencode_cfg,
                                                          const io::BaseListReader& base_list,
                                                          const io::IvfListsReader& ivf_lists,
                                                          const io::DatasetVectorReader* fallback_reader,
                                                          bool allow_random_fallback,
                                                          StreamKernelProvider* kernels,
                                                          const io::LinkageListStoreConfig& store_cfg,
                                                          LinkageDepthStats* base_depth_stats_out,
                                                          MseStats* base_linkage_mse_out,
                                                          std::string* err) {
        CpuStreamKernels cpu_kernels;
        if (!kernels) {
            kernels = &cpu_kernels;
        }
        // IMPORTANT: this stage is expected to be cluster-parallel.
        //
        // On some OpenMP runtimes (notably MSVC vcomp), earlier phases may leave the current thread's
        // nthreads-var at 1 (e.g. via helper routines that temporarily force single-threaded execution).
        // If we size TLS via `omp_get_max_threads()` and enter `#pragma omp parallel` without a
        // `num_threads(...)` clause, the whole linkage build can silently become single-threaded.
        //
        // Fix: derive a stable `nth` from config and force it on the outer parallel region.
        const int nth = std::max(1, (cfg.runtime.omp_threads > 0) ? cfg.runtime.omp_threads : omp_get_max_threads());
        if (cfg.runtime.omp_threads > 0) {
            omp_set_num_threads(nth);
        }
        const int nlist = ivf_lists.nlist();
        if (nlist <= 0) {
            if (err) *err = "BuildLinkageTwoCodebookVirtualStreamingByCluster: invalid nlist.";
            return false;
        }

        const int d = base_list.meta().d;
        const int m = cfg.model.m;
        const int m_codes = std::max(0, m - 1);
        if (d <= 0 || m <= 1) {
            if (err) *err = "BuildLinkageTwoCodebookVirtualStreamingByCluster: invalid d/m.";
            return false;
        }

        // Large pipeline: by default we avoid full-precomp (G(H×H)) for C_root/C_one and use the large-root CUDA
        // evaluator path. However, we still want the pipeline to be runnable without CUDA for smaller workloads.
        // In that case, we fall back to CPU evaluation by building full-precomp for C_one (needed for CPU eval).
        bool want_cuda_pool = false;
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
        want_cuda_pool = cfg.runtime.use_cuda && (cfg.runtime.cuda_pool_size > 0);
#endif
        const bool cpu_fallback_large = cfg.large.enabled && !want_cuda_pool;
        if (cpu_fallback_large) {
            LogWarn(
                "BuildLinkageTwoCodebookVirtualStreamingByCluster: large.enabled=true but CUDA pool is disabled; "
                "falling back to CPU candidate-eval with full-precomp(C_one). This may be slower.");
        }

        Precomp pre_root;
        if (cfg.large.enabled) {
            // Large pipeline default: do not build full-precomp (G(H×H)) for C_root.
            // CPU-only fallback: if virtual nodes are enabled, CPU UMAP-center encoding requires full-precomp(G).
            // This choice is controlled externally via config (virtual.enabled / runtime.use_cuda).
            const bool want_full_precomp_root = cpu_fallback_large && cfg.virtual_cfg.enabled;
            if (want_full_precomp_root) {
                if (!BuildPrecomp(train.C_root, &pre_root)) {
                    if (err) {
                        *err =
                            "BuildLinkageTwoCodebookVirtualStreamingByCluster: failed BuildPrecomp(C_root) for CPU-only large fallback.";
                    }
                    return false;
                }
            }
            else {
                pre_root = BuildPrecompMetaOnly(train.C_root);
            }
        }
        else {
            if (!BuildPrecomp(train.C_root, &pre_root)) {
                if (err) *err = "BuildLinkageTwoCodebookVirtualStreamingByCluster: failed BuildPrecomp(C_root).";
                return false;
            }
        }
        Precomp pre_one;
        if (cfg.large.enabled) {
            // Large pipeline: by default avoid full-precomp (G(H×H)) for C_one as well and use large-root CUDA evaluators.
            //
            // However, when the tiny-CPU mixed path is enabled, we must retain the CPU evaluator semantics for the
            // forced same-layer dynamic flush tail; the existing CPU evaluator uses Precomp::G.
            const bool want_tiny_cpu =
                cfg.runtime.cuda_linkage_same_dynamic_tiny_cpu &&
                (cfg.runtime.cuda_linkage_same_dynamic_tiny_cpu_max_pairs > 0);
            if (want_tiny_cpu || cpu_fallback_large) {
                if (!BuildPrecomp(train.C_one, &pre_one)) {
                    if (err)
                        *err =
                            "BuildLinkageTwoCodebookVirtualStreamingByCluster: failed BuildPrecomp(C_one) for CPU fallback/tiny-CPU path.";
                    return false;
                }
            }
            else {
                pre_one = BuildPrecompMetaOnly(train.C_one);
            }
        }
        else {
            if (!BuildPrecomp(train.C_one, &pre_one)) {
                if (err) *err = "BuildLinkageTwoCodebookVirtualStreamingByCluster: failed BuildPrecomp(C_one).";
                return false;
            }
        }
        if (train.C_one.books.empty() || train.C_one.books[0].cols > 256) {
            if (err) {
                *err = "BuildLinkageTwoCodebookVirtualStreamingByCluster: invalid C_one[0] size for large pipeline "
                    "(code0_one stored as uint8 requires h0_one<=256).";
            }
            return false;
        }
        if (pre_one.h_vec.empty() || pre_one.h_vec[0] != train.C_one.books[0].cols) {
            if (err) {
                *err = "BuildLinkageTwoCodebookVirtualStreamingByCluster: C_one precomp mismatch "
                    "(pre_one.h_vec[0] != C_one.books[0].cols).";
            }
            return false;
        }

        // When we do not build full-precomp for C_one (meta-only), we must rely on the large-root CUDA evaluator,
        // which requires device R_full gather (no pinned_Rp fallback is available without Precomp::C_all/G).
        if (cfg.large.enabled && pre_one.G.data.empty() && !cfg.runtime.cuda_linkage_use_device_rfull) {
            if (err) {
                *err =
                    "BuildLinkageTwoCodebookVirtualStreamingByCluster: large.enabled=true without full-precomp(C_one) "
                    "requires runtime.cuda_linkage_use_device_rfull=true.";
            }
            return false;
        }

        ColMajorMatrix<float> G_one_root;
        if (!pre_one.G.data.empty() && !pre_root.G.data.empty()) {
            ScopedBlasThreads blas_scope(std::max(1, omp_get_max_threads()));
            G_one_root = BuildOneRootCrossGram(pre_one, pre_root);
        }

        std::vector<bool> is_bad = train.is_bad_cluster;
        if (static_cast<int>(is_bad.size()) < nlist) {
            is_bad.assign(static_cast<std::size_t>(nlist), false);
        }

        [[maybe_unused]] CudaStreamKernelsPool* cuda_pool_ptr = nullptr;

#if defined(STLQ_ENABLE_CUDA)
        std::unique_ptr<CudaStreamKernelsPool> cuda_pool;
        if (cfg.runtime.use_cuda) {
            const int pool_size_cfg = cfg.runtime.cuda_pool_size;
            if (pool_size_cfg > 0) {
                const int pool_size = pool_size_cfg;
                CudaPoolConfig pool_cfg;
                pool_cfg.device = cfg.runtime.cuda_device;
                pool_cfg.allow_tf32 = EffectiveCudaAllowTf32(cfg.runtime);
                cuda_pool = std::make_unique<CudaStreamKernelsPool>(pool_size, pool_cfg);
                cuda_pool_ptr = cuda_pool.get();
                // Avoid spamming identical CUDA linkage header logs on every global-R iteration / baseset pass.
                // We still apply the runtime setters on each call (they are cheap and keep semantics obvious).
                {
                    static std::atomic<bool> did_log{false};
                    const bool effective_ils =
                        cfg.base.linkage.use_ils && cfg.base.linkage.ils_rounds > 0 && cfg.base.linkage.
                        ils_perturb_layers > 0;
                    bool expected = false;
                    if (did_log.compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
                        LogInfo("CUDA linkage: enabled (pool_size=" + std::to_string(pool_size) +
                            ", min_kp=" + std::to_string(cfg.runtime.cuda_linkage_min_kp) +
                            ", max_kp=" + std::to_string(cfg.runtime.cuda_linkage_max_kp) + ")");
                        if (effective_ils) {
                            LogInfo("CUDA linkage note: effective ILS is enabled => GPU candidate eval uses ILS+ICM.");
                        }
                    }
                }
                SetCudaLinkageEvalChunkMaxPairs(cfg.runtime.cuda_linkage_chunk_max_pairs);
                SetCudaLinkageEvalMemBudgetMb(cfg.runtime.cuda_linkage_mem_budget_mb);
                SetCudaLinkageEvalAsyncPinnedBudgetMb(cfg.runtime.cuda_linkage_eval_async_pinned_mb);
                SetCudaLinkageEncodeLargeRootXc0ChunkMb(cfg.runtime.cuda_linkage_large_root_xc0_chunk_mb);
                SetCudaLinkageEncodeLargeRootWorkspaceMb(cfg.runtime.cuda_linkage_mem_budget_mb);
            }
            else {
                LogInfo(
                    "CUDA linkage: disabled (runtime.cuda_pool_size<=0): forcing CPU candidate-eval for linkage build.");
            }
        }

        struct ScopedCudaLinkageEvalProfiling
        {
            explicit ScopedCudaLinkageEvalProfiling(bool enabled_) : enabled(enabled_) {
                if (enabled) {
                    SetCudaLinkageEvalProfiling(true);
                }
            }

            ~ScopedCudaLinkageEvalProfiling() {
                if (enabled) {
                    SetCudaLinkageEvalProfiling(false);
                }
            }

            bool enabled = false;
        };
        const bool cuda_eval_profile = (cuda_pool_ptr != nullptr) && cfg.large.profile_timing;
        ScopedCudaLinkageEvalProfiling cuda_eval_profile_guard(cuda_eval_profile);
#endif

        const LinkageBuildConfig& linkage_cfg = cfg.base.linkage;
        // `base_list` may contain raw vectors aligned with list-order:
        // - raw_f32.bin for float datasets (preferred, lossless)
        // - raw_u8.bin for u8 datasets
        //
        // IMPORTANT: do NOT mix dtypes. For f32 datasets, never consume raw_u8.bin (legacy clamped output).
        // For u8 datasets, never consume raw_f32.bin.
        const bool has_raw_u8 = base_list.HasRawU8();
        const bool has_raw_f32 = base_list.HasRawF32();

        const bool profile_timing = cfg.large.profile_timing;
        auto wall_now_s = []() -> double
        {
            using clock = std::chrono::steady_clock;
            return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
        };
        double t_wall_pass_a = 0.0;
        double t_wall_coeff_pre = 0.0;
        double t_wall_pass_b = 0.0;
        double t_wall_writer_finish = 0.0;
        double t_wall_coeff_finish = 0.0;
        double t_wall_reduce = 0.0;
        double t_wall_t0 = profile_timing ? wall_now_s() : 0.0;

        // Pass A: precompute per-cluster sizes/prefix sums for the linkage_list store and open a positioned writer.
        io::LinkageListStoreConfig out_cfg = store_cfg;
        out_cfg.nlist = nlist;
        out_cfg.m_codes = m_codes;
        // code0_one (C_one[0]) is stored as uint8 in the large-scale pipeline.
        out_cfg.code0_width_bytes = 1;
        // Parent representation is caller-controlled:
        // - baseset disk pipeline may prefer LOUDS-only to save disk
        // - train/iteration linkage stages typically keep parent.u32 and do not need LOUDS
        out_cfg.store_parent_louds = store_cfg.store_parent_louds;
        out_cfg.store_parent_u32 = store_cfg.store_parent_u32;
        if (out_cfg.store_parent_louds) {
            out_cfg.parent_louds_select_stride = std::max(1, cfg.eval.parent_louds_select_stride);
            out_cfg.parent_louds_rank_words_per_super_log2 =
                std::max(1, cfg.eval.parent_louds_rank_words_per_super_log2);
        }
        if (out_cfg.dir.empty()) {
            if (err) *err = "BuildLinkageTwoCodebookVirtualStreamingByCluster: empty linkage_list dir.";
            return false;
        }

        const bool use_coeff_codec = cfg.large.linkage_coeff_codec.enabled;
        const bool store_coeffs_f32 = out_cfg.store_coeffs_f32;

        // depth_offsets is stored with a fixed length so Pass A does not depend on realized linkage depth.
        const std::uint32_t fixed_depth_len =
            static_cast<std::uint32_t>(std::max(2, std::max(0, linkage_cfg.max_depth) + 2));

        io::LinkageListWritePlan plan;
        plan.nlist = nlist;
        plan.m_codes = m_codes;
        plan.small_code_width_bytes = io::LinkageListStoreConfig::kSmallCodeWidthBytes;
        plan.code0_width_bytes = out_cfg.code0_width_bytes;
        plan.real_offsets.assign(static_cast<std::size_t>(nlist) + 1, 0);
        plan.virt_offsets.assign(static_cast<std::size_t>(nlist) + 1, 0);
        plan.depth_offsets_offsets.assign(static_cast<std::size_t>(nlist) + 1, 0);
        plan.parent_louds_offsets_bytes.assign(static_cast<std::size_t>(nlist) + 1, 0);
        plan.n_real.assign(static_cast<std::size_t>(nlist), 0);
        plan.n_virt.assign(static_cast<std::size_t>(nlist), 0);
        plan.depth_len.assign(static_cast<std::size_t>(nlist), fixed_depth_len);

        for (int cid = 0; cid < nlist; ++cid) {
            const std::uint32_t n_real = ivf_lists.ListSize(cid);
            plan.n_real[static_cast<std::size_t>(cid)] = n_real;
            std::uint32_t n_virt = 0;
            if (cfg.virtual_cfg.enabled && is_bad[static_cast<std::size_t>(cid)] && n_real > 1) {
                n_virt = static_cast<std::uint32_t>(ResolveVirtualCount(static_cast<int>(n_real), cfg.virtual_cfg));
            }
            plan.n_virt[static_cast<std::size_t>(cid)] = n_virt;
            plan.real_offsets[static_cast<std::size_t>(cid) + 1] =
                plan.real_offsets[static_cast<std::size_t>(cid)] + static_cast<std::uint64_t>(n_real);
            plan.virt_offsets[static_cast<std::size_t>(cid) + 1] =
                plan.virt_offsets[static_cast<std::size_t>(cid)] + static_cast<std::uint64_t>(n_virt);
            plan.depth_offsets_offsets[static_cast<std::size_t>(cid) + 1] =
                plan.depth_offsets_offsets[static_cast<std::size_t>(cid)] + static_cast<std::uint64_t>(fixed_depth_len);
            const std::uint64_t n_total = static_cast<std::uint64_t>(n_real) + static_cast<std::uint64_t>(n_virt);
            const std::uint64_t blob_bytes =
                stlq::succinct::ParentLOUDS::SerializedBytesForNodes(static_cast<std::uint32_t>(n_total));
            plan.parent_louds_offsets_bytes[static_cast<std::size_t>(cid) + 1] =
                plan.parent_louds_offsets_bytes[static_cast<std::size_t>(cid)] + blob_bytes;
        }

        plan.total_depth_offsets_u32_bytes =
            plan.depth_offsets_offsets.back() * static_cast<std::uint64_t>(sizeof(std::uint32_t));
        plan.total_real_u32_bytes = plan.real_offsets.back() * static_cast<std::uint64_t>(sizeof(std::uint32_t));
        plan.total_parent_u32_bytes = out_cfg.store_parent_u32
                                          ? plan.real_offsets.back() * static_cast<std::uint64_t>(sizeof(std::uint32_t))
                                          : 0u;
        plan.total_codes_bytes =
            plan.real_offsets.back() * static_cast<std::uint64_t>(m_codes) *
            static_cast<std::uint64_t>(io::LinkageListStoreConfig::kSmallCodeWidthBytes);
        plan.total_coeffs_f32_bytes = store_coeffs_f32
                                          ? plan.real_offsets.back() * static_cast<std::uint64_t>(m_codes) *
                                          static_cast<std::uint64_t>(sizeof(float))
                                          : 0;
        plan.total_code0_one_bytes =
            plan.real_offsets.back() * static_cast<std::uint64_t>(out_cfg.code0_width_bytes);
        plan.total_a0_f32_bytes = store_coeffs_f32
                                      ? plan.real_offsets.back() * static_cast<std::uint64_t>(sizeof(float))
                                      : 0;
        plan.total_virt_codes_bytes =
            plan.virt_offsets.back() * static_cast<std::uint64_t>(m_codes) *
            static_cast<std::uint64_t>(io::LinkageListStoreConfig::kSmallCodeWidthBytes);
        plan.total_virt_coeffs_f32_bytes = store_coeffs_f32
                                               ? plan.virt_offsets.back() * static_cast<std::uint64_t>(m_codes) *
                                               static_cast<std::uint64_t>(sizeof(float))
                                               : 0;
        plan.total_virt_a0_f32_bytes = store_coeffs_f32
                                           ? plan.virt_offsets.back() * static_cast<std::uint64_t>(sizeof(float))
                                           : 0;
        plan.total_parent_louds_bytes = plan.parent_louds_offsets_bytes.back();

        // ---- Cluster-level checkpoint/resume (baseset linkage stage) ----
        // Rationale: OpenMP cluster completion is out-of-order; a sequential "last cid" checkpoint is unsafe.
        // We persist a per-cluster done marker and per-cluster summary stats. On resume, we skip done clusters.
        const bool use_linkage_ckpt = cfg.large.enabled && cfg.large.linkage_checkpoint && out_cfg.enable_checkpoint;
        ClusterCheckpoint ckpt;
        bool resume_writers = false;
        if (use_linkage_ckpt) {
            std::string ckpt_err;
            if (!ckpt.Open(out_cfg.dir, nlist, &ckpt_err)) {
                if (err) *err = "BuildLinkageTwoCodebookVirtualStreamingByCluster: checkpoint open failed: " + ckpt_err;
                return false;
            }
            // If the done file already existed (or any cluster is marked done), we must NOT truncate outputs.
            resume_writers = std::any_of(ckpt.done.begin(), ckpt.done.end(), [](std::uint8_t v) { return v != 0; });
            if (resume_writers) {
                LogInfo("Linkage checkpoint enabled: resuming from existing per-cluster markers.");
            }
            else {
                LogInfo("Linkage checkpoint enabled: starting a fresh run (checkpoint files created).");
            }
        }

        io::LinkageListRandomWriter random_writer;
        if (resume_writers) {
            if (!random_writer.OpenResume(out_cfg, plan, err)) {
                return false;
            }
        }
        else {
            if (!random_writer.Open(out_cfg, plan, err)) {
                return false;
            }
        }

        if (profile_timing) {
            const double t1 = wall_now_s();
            t_wall_pass_a = t1 - t_wall_t0;
            t_wall_t0 = t1;
        }

        io::LinkageCoeffCodecRandomWriter coeff_writer;
        io::LinkageCoeffCodecStats coeff_stats{};
        const io::LinkageCoeffCodecMeta* coeff_meta = nullptr;
        std::vector<float> coeff_scales_ones;
        std::vector<std::vector<std::uint8_t>> coeff_lens_zeros;
        std::vector<std::vector<std::uint8_t>> coeff_payload_empty;
        if (use_coeff_codec) {
            io::LinkageCoeffCodecStoreConfig ccfg;
            ccfg.dir = out_cfg.dir;
            ccfg.nlist = nlist;
            ccfg.m = m;
            ccfg.granularity = cfg.large.linkage_coeff_codec.granularity;
            int max_bits = 2;
            for (int b : cfg.large.linkage_coeff_codec.bits_per_layer) {
                max_bits = std::max(max_bits, b);
            }
            ccfg.max_bits = max_bits;
            ccfg.alphabet_size = 0; // derive from max_bits
            if (resume_writers) {
                if (!coeff_writer.OpenResume(ccfg, err)) {
                    return false;
                }
            }
            else {
                if (!coeff_writer.Open(ccfg, err)) {
                    return false;
                }
            }
            coeff_stats = coeff_writer.stats();
            coeff_meta = &coeff_writer.meta();
            coeff_scales_ones.assign(static_cast<std::size_t>(m), 1.0f);
            coeff_lens_zeros.assign(static_cast<std::size_t>(coeff_meta->streams),
                                    std::vector<std::uint8_t>(static_cast<std::size_t>(coeff_meta->S), 0));
            coeff_payload_empty.assign(static_cast<std::size_t>(coeff_meta->streams), {});
        }

        struct CoeffCodecReconPrecomp
        {
            int d = 0;
            int m = 0;

            // Root pack.
            std::vector<int> h_root;
            std::vector<int> root_small_offsets; // size m, offsets for layers 1..m-1 into C_root_small
            int H_root_small = 0;
            ColMajorMatrix<float> C_root_small; // d×H_root_small
            ColMajorMatrix<float> G_root_small; // H_root_small×H_root_small
            std::vector<float> norm0_root; // ||C_root[0][:,cid]||^2

            // One pack.
            std::vector<int> h_one;
            std::vector<int> one_offsets; // size m, offsets into flattened C_one_all
            int H_one_total = 0;
            ColMajorMatrix<float> G_one_all; // H_one_total×H_one_total

            // 1-based norm2 tables for weighted quantile.
            std::vector<std::vector<float>> norm2_root_layers; // size m, each (h+1)
            std::vector<std::vector<float>> norm2_one_layers; // size m, each (h+1)
            std::vector<Span<const float>> norm2_root_spans;
            std::vector<Span<const float>> norm2_one_spans;
        };

        std::unique_ptr<CoeffCodecReconPrecomp> coeff_pre;
        if (use_coeff_codec) {
            coeff_pre = std::make_unique<CoeffCodecReconPrecomp>();
            coeff_pre->d = d;
            coeff_pre->m = m;

            auto norm2_col = [&](const float* x) -> float
            {
                float s = 0.0f;
#pragma omp simd reduction(+:s)
                for (int i = 0; i < d; ++i) {
                    s += x[i] * x[i];
                }
                return s;
            };

            // Root pack small layers concat + small-small gram.
            coeff_pre->h_root.resize(static_cast<std::size_t>(m), 0);
            for (int l = 0; l < m; ++l) {
                coeff_pre->h_root[static_cast<std::size_t>(l)] = train.C_root.books[static_cast<std::size_t>(l)].cols;
            }
            coeff_pre->root_small_offsets.assign(static_cast<std::size_t>(m), 0);
            int Hs = 0;
            for (int l = 1; l < m; ++l) {
                coeff_pre->root_small_offsets[static_cast<std::size_t>(l)] = Hs;
                Hs += coeff_pre->h_root[static_cast<std::size_t>(l)];
            }
            coeff_pre->H_root_small = Hs;
            coeff_pre->C_root_small = ColMajorMatrix<float>(d, Hs);
            {
                int col = 0;
                for (int l = 1; l < m; ++l) {
                    const auto& book = train.C_root.books[static_cast<std::size_t>(l)];
                    for (int c = 0; c < book.cols; ++c) {
                        std::memcpy(coeff_pre->C_root_small.Col(col + c), book.Col(c),
                                    sizeof(float) * static_cast<std::size_t>(d));
                    }
                    col += book.cols;
                }
            }
            coeff_pre->G_root_small = ColMajorMatrix<float>(Hs, Hs);
            {
                ScopedBlasThreads blas_scope(std::max(1, omp_get_max_threads()));
                Gemm(true, false, 1.0f, coeff_pre->C_root_small, coeff_pre->C_root_small, 0.0f,
                     &coeff_pre->G_root_small);
            }

            // Root norms for layer0 (by cluster id).
            const int h0 = coeff_pre->h_root[0];
            coeff_pre->norm0_root.assign(static_cast<std::size_t>(std::max(0, h0)), 0.0f);
            for (int cid = 0; cid < h0; ++cid) {
                coeff_pre->norm0_root[static_cast<std::size_t>(cid)] =
                    norm2_col(train.C_root.books[0].Col(cid));
            }

            // One pack gram (all layers).
            coeff_pre->h_one.resize(static_cast<std::size_t>(m), 0);
            coeff_pre->one_offsets.assign(static_cast<std::size_t>(m), 0);
            int Ho = 0;
            for (int l = 0; l < m; ++l) {
                coeff_pre->one_offsets[static_cast<std::size_t>(l)] = Ho;
                coeff_pre->h_one[static_cast<std::size_t>(l)] = train.C_one.books[static_cast<std::size_t>(l)].cols;
                Ho += coeff_pre->h_one[static_cast<std::size_t>(l)];
            }
            coeff_pre->H_one_total = Ho;
            ColMajorMatrix<float> C_one_all(d, Ho);
            {
                int col = 0;
                for (int l = 0; l < m; ++l) {
                    const auto& book = train.C_one.books[static_cast<std::size_t>(l)];
                    for (int c = 0; c < book.cols; ++c) {
                        std::memcpy(C_one_all.Col(col + c), book.Col(c),
                                    sizeof(float) * static_cast<std::size_t>(d));
                    }
                    col += book.cols;
                }
            }
            coeff_pre->G_one_all = ColMajorMatrix<float>(Ho, Ho);
            {
                ScopedBlasThreads blas_scope(std::max(1, omp_get_max_threads()));
                Gemm(true, false, 1.0f, C_one_all, C_one_all, 0.0f, &coeff_pre->G_one_all);
            }

            // 1-based norm2 tables.
            coeff_pre->norm2_root_layers.resize(static_cast<std::size_t>(m));
            coeff_pre->norm2_one_layers.resize(static_cast<std::size_t>(m));
            for (int l = 0; l < m; ++l) {
                const auto& br = train.C_root.books[static_cast<std::size_t>(l)];
                auto& nr = coeff_pre->norm2_root_layers[static_cast<std::size_t>(l)];
                nr.assign(static_cast<std::size_t>(br.cols) + 1u, 0.0f);
                for (int c = 0; c < br.cols; ++c) {
                    nr[static_cast<std::size_t>(c + 1)] = norm2_col(br.Col(c));
                }
                const auto& bo = train.C_one.books[static_cast<std::size_t>(l)];
                auto& no = coeff_pre->norm2_one_layers[static_cast<std::size_t>(l)];
                no.assign(static_cast<std::size_t>(bo.cols) + 1u, 0.0f);
                for (int c = 0; c < bo.cols; ++c) {
                    no[static_cast<std::size_t>(c + 1)] = norm2_col(bo.Col(c));
                }
            }
            coeff_pre->norm2_root_spans.clear();
            coeff_pre->norm2_one_spans.clear();
            coeff_pre->norm2_root_spans.reserve(static_cast<std::size_t>(m));
            coeff_pre->norm2_one_spans.reserve(static_cast<std::size_t>(m));
            for (int l = 0; l < m; ++l) {
                const auto& nr = coeff_pre->norm2_root_layers[static_cast<std::size_t>(l)];
                const auto& no = coeff_pre->norm2_one_layers[static_cast<std::size_t>(l)];
                coeff_pre->norm2_root_spans.emplace_back(nr.data(), nr.size());
                coeff_pre->norm2_one_spans.emplace_back(no.data(), no.size());
            }
        }

        if (profile_timing) {
            const double t1 = wall_now_s();
            t_wall_coeff_pre = t1 - t_wall_t0;
            t_wall_t0 = t1;
        }

        // Pass B: cluster-parallel linkage build + positioned write.
        std::vector<std::uint64_t> cluster_real(static_cast<std::size_t>(nlist), 0);
        std::vector<std::uint64_t> cluster_linkaged(static_cast<std::size_t>(nlist), 0);
        std::vector<double> cluster_depth_sum(static_cast<std::size_t>(nlist), 0.0);
        std::vector<int> cluster_max_depth(static_cast<std::size_t>(nlist), 0);
        std::vector<double> cluster_mse_sum(static_cast<std::size_t>(nlist), 0.0);
        std::vector<double> cluster_mse_min(static_cast<std::size_t>(nlist), std::numeric_limits<double>::infinity());
        std::vector<double> cluster_mse_max(static_cast<std::size_t>(nlist), 0.0);

        if (use_linkage_ckpt && resume_writers) {
            std::string ckpt_err;
            if (!ckpt.LoadStatsInto(&cluster_real,
                                    &cluster_linkaged,
                                    &cluster_depth_sum,
                                    &cluster_max_depth,
                                    &cluster_mse_sum,
                                    &cluster_mse_min,
                                    &cluster_mse_max,
                                    &ckpt_err)) {
                if (err) {
                    *err = "BuildLinkageTwoCodebookVirtualStreamingByCluster: checkpoint stats load failed: " +
                        ckpt_err;
                }
                return false;
            }
        }

        struct LinkageStreamingTlsTiming
        {
            double read_ivf = 0.0;
            double read_raw = 0.0;
            double read_base = 0.0;
            // Time spent blocked waiting for the async IO prefetch queue (not included in read_*).
            double wait_async_pop = 0.0;
            double bad_knn_umap = 0.0;
            double linkage_core = 0.0;
            double write = 0.0;
            stlq::linkage::LinkageBuildProfileTls prof;
        };
        std::vector<LinkageStreamingTlsTiming> timing_tls(static_cast<std::size_t>(nth));

        std::atomic<int> done{0};
        std::atomic<long long> last_print_ns{0};
        std::atomic<bool> ok{true};
        std::string first_err;
        std::mutex err_mu;

        auto now_ns = []() -> long long
        {
            using clock = std::chrono::steady_clock;
            return std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now().time_since_epoch()).count();
        };
        FILE* progress_stream = stderr;

        const bool kernels_is_cpu = (dynamic_cast<CpuStreamKernels*>(kernels) != nullptr);
        std::vector<CpuStreamKernels> cpu_kernels_tls;
        if (kernels_is_cpu) {
            cpu_kernels_tls.resize(static_cast<std::size_t>(nth));
        }

        // Snapshot of clusters already done (resume-only). Used to skip async prefetch IO.
        std::shared_ptr<std::vector<std::uint8_t>> resume_done_snapshot;
        int resume_done_count = 0;
        if (use_linkage_ckpt && resume_writers) {
            resume_done_snapshot = std::make_shared<std::vector<std::uint8_t>>(ckpt.done);
            resume_done_count = static_cast<int>(std::count_if(resume_done_snapshot->begin(),
                                                               resume_done_snapshot->end(),
                                                               [](std::uint8_t v) { return v != 0; }));
        }

        const bool can_async_io =
            cfg.runtime.linkage_async_io && cfg.runtime.linkage_async_io_depth > 0 &&
            (has_raw_f32 || has_raw_u8);
        std::unique_ptr<AsyncClusterPrefetcher> async_io;
        if (can_async_io) {
            async_io = std::make_unique<AsyncClusterPrefetcher>(base_list,
                                                                ivf_lists,
                                                                nlist,
                                                                resume_done_snapshot
                                                                    ? resume_done_snapshot.get()
                                                                    : nullptr,
                                                                has_raw_f32,
                                                                has_raw_u8,
                                                                cfg.runtime.linkage_async_io_depth,
                                                                profile_timing,
                                                                &ok,
                                                                &first_err,
                                                                &err_mu,
                                                                "BuildLinkageTwoCodebookVirtualStreamingByCluster/async_io");
            async_io->Start();
        }

        // If async prefetcher skips already-done clusters, seed progress counter accordingly.
        if (async_io && resume_done_count > 0) {
            done.store(resume_done_count);
        }

        std::atomic<int> next_cid{0};

#pragma omp parallel num_threads(nth) default(none) shared(timing_tls, cpu_kernels_tls, done, last_print_ns, ok, first_err, err_mu, next_cid, async_io, ckpt, base_list, ivf_lists, train, cfg, linkage_cfg, umap_reencode_cfg, pre_root, pre_one, G_one_root, is_bad, fallback_reader, random_writer, coeff_writer, coeff_meta, coeff_pre, coeff_scales_ones, coeff_lens_zeros, coeff_payload_empty, plan, out_cfg, cluster_real, cluster_linkaged, cluster_depth_sum, cluster_max_depth, cluster_mse_sum, cluster_mse_min, cluster_mse_max, kernels, cuda_pool_ptr) firstprivate(now_ns, progress_stream, d, m, m_codes, nlist, fixed_depth_len, allow_random_fallback, profile_timing, kernels_is_cpu, has_raw_u8, has_raw_f32, use_coeff_codec, use_linkage_ckpt)
        {
            ClusterWorkBuf buf;
            const int tid = omp_get_thread_num();
            LinkageStreamingTlsTiming& t_time = timing_tls[static_cast<std::size_t>(tid)];
            stlq::linkage::SetLinkageBuildProfileTls(profile_timing ? &t_time.prof : nullptr);
            StreamKernelProvider* kernels_local =
                kernels_is_cpu
                    ? static_cast<StreamKernelProvider*>(&cpu_kernels_tls[static_cast<std::size_t>(tid)])
                    : kernels;
            std::string local_err;
            io::BaseListThreadReader base_thr;
            io::IvfListsThreadReader ivf_thr;
            const bool use_async_io = (async_io != nullptr);
            auto now_s = []() -> double
            {
                using clock = std::chrono::steady_clock;
                return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
            };
            auto fail = [&](const std::string& msg)
            {
                ok.store(false);
                std::lock_guard<std::mutex> guard(err_mu);
                if (first_err.empty()) {
                    first_err = msg.empty() ? "BuildLinkageTwoCodebookVirtualStreamingByCluster: failed." : msg;
                }
            };

            auto bump_progress = [&]()
            {
                const int cur_done = done.fetch_add(1) + 1;
                // Progress printing: allow any thread to print, but throttle via CAS on `last_print_ns`.
                // Avoid printing the final (nlist) progress here; a single final line is printed after the parallel region.
                if (cur_done < nlist) {
                    const long long now = now_ns();
                    long long last = last_print_ns.load(std::memory_order_relaxed);
                    const bool want_dense = (cur_done <= 16) || ((cur_done % 16) == 0);
                    const bool want_time = (last == 0) || (now - last >= 1000000000LL);
                    if ((want_dense || want_time) &&
                        last_print_ns.compare_exchange_strong(last, now, std::memory_order_relaxed)) {
                        const int shown = done.load(std::memory_order_relaxed);
                        std::fprintf(progress_stream, "\rLinkage build: %d/%d clusters   ", shown, nlist);
                        std::fflush(progress_stream);
                    }
                }
            };

            if (!use_async_io) {
                if (!base_thr.OpenFrom(base_list, &local_err)) {
                    fail(local_err);
                }
                if (!ivf_thr.OpenFrom(ivf_lists, &local_err)) {
                    fail(local_err);
                }
            }

            PrefetchedClusterSpans item;
            while (true) {
                if (!ok.load(std::memory_order_relaxed)) {
                    break;
                }
                int cid = -1;
                std::uint64_t off = 0;
                local_err.clear();
                buf.ClearPerCluster();
                auto& ids = buf.ids;
                if (use_async_io) {
                    const double t0_pop = profile_timing ? now_s() : 0.0;
                    if (!async_io->Pop(&item)) {
                        break;
                    }
                    if (profile_timing) {
                        t_time.wait_async_pop += now_s() - t0_pop;
                    }
                    cid = item.cid;
                    off = item.off;
                    if (use_linkage_ckpt && ckpt.IsDone(cid)) {
                        bump_progress();
                        continue;
                    }
                    ids = std::move(item.ids);
                }
                else {
                    cid = next_cid.fetch_add(1);
                    if (cid >= nlist) {
                        break;
                    }
                    if (use_linkage_ckpt && ckpt.IsDone(cid)) {
                        bump_progress();
                        continue;
                    }
                    const double t0_ivf = profile_timing ? now_s() : 0.0;
                    if (!ivf_thr.ReadList(ivf_lists, cid, &ids, &local_err)) {
                        fail(local_err);
                        continue;
                    }
                    if (profile_timing) {
                        t_time.read_ivf += now_s() - t0_ivf;
                    }
                    off = ivf_lists.Offset(cid);
                }
                const int n_real = static_cast<int>(ids.size());

                // Read raw vectors in list-order if available; otherwise allow a slow random-IO fallback.
                const double t0_raw = profile_timing ? now_s() : 0.0;
                ColMajorMatrix<float>& Xrot = buf.Xrot;
                if (n_real > 0) {
                    if (use_async_io) {
                        // Async IO already materialized raw; only ensure Xrot (avoid resizing other buffers).
                        Xrot.rows = d;
                        Xrot.cols = n_real;
                        const std::size_t need = static_cast<std::size_t>(d) * static_cast<std::size_t>(n_real);
                        if (Xrot.data.size() < need) {
                            Xrot.data.resize(need);
                        }
                        if (has_raw_f32) {
                            buf.x_f32 = std::move(item.x_f32);
                            kernels_local->Gemm(false, false, 1.0f, train.R, buf.x_f32, 0.0f, &Xrot);
                        }
                        else if (has_raw_u8) {
                            buf.x_u8 = std::move(item.x_u8);
                            kernels_local->ConvertU8ToF32AndRotate(buf.x_u8, train.R, &Xrot);
                        }
                        else {
                            fail("BuildLinkageTwoCodebookVirtualStreamingByCluster: async IO requires list-order raw.");
                            continue;
                        }
                    }
                    else {
                        buf.EnsureReal(d, n_real, m, m_codes);
                        if (ReaderCanReadF32(fallback_reader)) {
                            // f32 dataset: only raw_f32 (preferred) or fvecs/fbin fallback are valid.
                            if (has_raw_f32) {
                                ColMajorMatrix<float>& x_f32 = buf.x_f32;
                                if (!base_thr.ReadRawF32Span(off, static_cast<std::uint32_t>(n_real), &x_f32,
                                                             &local_err)) {
                                    fail(local_err);
                                    continue;
                                }
                                kernels_local->Gemm(false, false, 1.0f, train.R, x_f32, 0.0f, &Xrot);
                            }
                            else {
                                ColMajorMatrix<float> x_f32;
                                const bool rok = io::ReadDatasetVectorByIdsF32(
                                    *fallback_reader, ids, &x_f32, &local_err);
                                if (!rok) {
                                    fail(local_err);
                                    continue;
                                }
                                Xrot.rows = train.R.rows;
                                Xrot.cols = x_f32.cols;
                                const std::size_t need =
                                    static_cast<std::size_t>(Xrot.rows) * static_cast<std::size_t>(Xrot.cols);
                                if (Xrot.data.size() < need) {
                                    Xrot.data.resize(need);
                                }
                                kernels_local->Gemm(false, false, 1.0f, train.R, x_f32, 0.0f, &Xrot);
                            }
                        }
                        else if (ReaderCanReadU8(fallback_reader)) {
                            // u8 dataset: only raw_u8 (preferred) or bvecs fallback are valid.
                            if (has_raw_u8) {
                                ColMajorMatrix<std::uint8_t>& x_u8 = buf.x_u8;
                                if (!base_thr.
                                    ReadRawU8Span(off, static_cast<std::uint32_t>(n_real), &x_u8, &local_err)) {
                                    fail(local_err);
                                    continue;
                                }
                                kernels_local->ConvertU8ToF32AndRotate(x_u8, train.R, &Xrot);
                            }
                            else {
                                ColMajorMatrix<std::uint8_t>& x_u8 = buf.x_u8;
                                if (!io::ReadDatasetVectorByIdsU8(*fallback_reader, ids, &x_u8, &local_err)) {
                                    fail(local_err);
                                    continue;
                                }
                                kernels_local->ConvertU8ToF32AndRotate(x_u8, train.R, &Xrot);
                            }
                        }
                        else {
                            // No dataset reader available: only allow using list-order raw (if present).
                            if (has_raw_f32) {
                                ColMajorMatrix<float>& x_f32 = buf.x_f32;
                                if (!base_thr.ReadRawF32Span(off, static_cast<std::uint32_t>(n_real), &x_f32,
                                                             &local_err)) {
                                    fail(local_err);
                                    continue;
                                }
                                kernels_local->Gemm(false, false, 1.0f, train.R, x_f32, 0.0f, &Xrot);
                            }
                            else if (has_raw_u8) {
                                ColMajorMatrix<std::uint8_t>& x_u8 = buf.x_u8;
                                if (!base_thr.
                                    ReadRawU8Span(off, static_cast<std::uint32_t>(n_real), &x_u8, &local_err)) {
                                    fail(local_err);
                                    continue;
                                }
                                kernels_local->ConvertU8ToF32AndRotate(x_u8, train.R, &Xrot);
                            }
                            else {
                                if (!allow_random_fallback) {
                                    fail(
                                        "BuildLinkageTwoCodebookVirtualStreamingByCluster: raw vectors missing and random fallback is disabled.");
                                    continue;
                                }
                                fail("BuildLinkageTwoCodebookVirtualStreamingByCluster: no fallback reader provided.");
                                continue;
                            }
                        }
                    }
                }
                if (profile_timing) {
                    t_time.read_raw += now_s() - t0_raw;
                }

                // Read base encodings (list-order aligned with IVF lists).
                const double t0_base = profile_timing ? now_s() : 0.0;
                ColMajorMatrix<Code>& B_small = buf.B_small;
                ColMajorMatrix<float>& a = buf.a;
                if (n_real > 0) {
                    if (use_async_io) {
                        B_small.data = std::move(item.codes);
                        a.data = std::move(item.coeffs);
                    }
                    else {
                        if (!base_thr.ReadCodesSpan(off, static_cast<std::uint32_t>(n_real), &B_small.data,
                                                    &local_err)) {
                            fail(local_err);
                            continue;
                        }
                        if (!base_thr.ReadCoeffsSpan(off, static_cast<std::uint32_t>(n_real), &a.data, &local_err)) {
                            fail(local_err);
                            continue;
                        }
                    }
                    B_small.rows = m_codes;
                    B_small.cols = n_real;
                    a.rows = m;
                    a.cols = n_real;
                    if (static_cast<int>(B_small.data.size()) != m_codes * n_real ||
                        static_cast<int>(a.data.size()) != m * n_real) {
                        fail("BuildLinkageTwoCodebookVirtualStreamingByCluster: base_list span size mismatch.");
                        continue;
                    }
                }
                else {
                    B_small.rows = m_codes;
                    B_small.cols = 0;
                    B_small.data.clear();
                    a.rows = m;
                    a.cols = 0;
                    a.data.clear();
                }
                if (profile_timing) {
                    t_time.read_base += now_s() - t0_base;
                }

                // Build full B (with root code in row0).
                ColMajorMatrix<FullCode>& B_full = buf.B_full;
                B_full.rows = m;
                B_full.cols = n_real;
                if (B_full.data.size() < static_cast<std::size_t>(m) * static_cast<std::size_t>(n_real)) {
                    B_full.data.resize(static_cast<std::size_t>(m) * static_cast<std::size_t>(n_real));
                }
                const bool need_root_u32 = (!train.C_root.books.empty() && train.C_root.books.front().cols > 256);
                for (int i = 0; i < n_real; ++i) {
                    // When h0_root>256, `FullCode` (u8) cannot represent `cid`. The forced root code is passed
                    // separately through the linkageing/eval path; keep row0 a dummy to avoid truncation.
                    B_full(0, i) = need_root_u32 ? static_cast<FullCode>(0) : static_cast<FullCode>(cid);
                    for (int l = 1; l < m; ++l) {
                        B_full(l, i) = B_small(l - 1, i);
                    }
                }

                LinkageStructure::Cluster& cluster_out = buf.cluster_out;
                int ls_failures = 0;
                MseStats cluster_linkage_mse{};
                double cluster_mse_sum_local = 0.0;
                ColMajorMatrix<FullCode>& B_virt = buf.B_virt;
                ColMajorMatrix<float>& a_virt = buf.a_virt;
                int n_virt = 0;

                cluster_out.cluster_id = cid;
                if (n_real <= 0) {
                    // Empty cluster: still write an empty record for this cid so downstream readers can
                    // rely on fixed-size per-cluster offsets. No linkageing is performed.
                    cluster_out.n_real = 0;
                    cluster_out.n_virtual = 0;
                    cluster_out.indices.clear();
                    cluster_out.parent_local.clear();
                    cluster_out.depth_offsets.assign(2, 0);
                }
                else if (cfg.virtual_cfg.enabled && is_bad[static_cast<std::size_t>(cid)] && n_real > 1) {
                    // Bad cluster: build kNN table once, run UMAP-like virtual roots, then multi-center linkageing.
                    const int mult_bad = std::max(1, cfg.hnsw.candidate_multiplier_bad);
                    int knn_real = std::max(std::max(1, cfg.virtual_cfg.umap_knn_k),
                                            mult_bad * std::max(1, linkage_cfg.knn_k));
                    knn_real = std::min(knn_real, n_real - 1);
                    const int knn_umap = std::min(std::max(1, cfg.virtual_cfg.umap_knn_k), knn_real);
                    const int k_virtual = ResolveVirtualCount(n_real, cfg.virtual_cfg);

                    std::vector<int>& cols_local = buf.cols_local;
                    cols_local.resize(static_cast<std::size_t>(n_real));
                    std::iota(cols_local.begin(), cols_local.end(), 0);

                    const double t0_bad = profile_timing ? now_s() : 0.0;
                    std::vector<std::uint32_t>& knn_ids_flat = buf.knn_ids_flat;
                    std::vector<float>& knn_dists_flat = buf.knn_dists_flat;
                    const double t0_bad_knn = stlq::linkage::LinkageBuildProfileEnabled()
                                                  ? stlq::linkage::LinkageBuildWallNowS()
                                                  : 0.0;
                    if (!BuildClusterKnnHnswImpl(Xrot, cols_local, cfg.hnsw, knn_real,
                                                 &knn_ids_flat, &knn_dists_flat, &local_err)) {
                        fail(local_err);
                        continue;
                    }
                    if (stlq::linkage::LinkageBuildProfileEnabled()) {
                        t_time.prof.bad_knn_s += stlq::linkage::LinkageBuildWallNowS() - t0_bad_knn;
                    }

                    ColMajorMatrix<float>& X_virt = buf.X_virt;
                    const double t0_bad_umap = stlq::linkage::LinkageBuildProfileEnabled()
                                                   ? stlq::linkage::LinkageBuildWallNowS()
                                                   : 0.0;
                    const auto seed =
                        static_cast<std::uint32_t>(umap_reencode_cfg.seed + 1337u * static_cast<std::uint32_t>(cid) +
                            17u);
                    if (!AddVirtualRootsUmapReencodeClusterFromKnnTables(cfg.virtual_cfg,
                                                                         train.C_root,
                                                                         pre_root,
#if defined(STLQ_ENABLE_CUDA)
                                                                         &cfg.runtime,
                                                                         cuda_pool_ptr,
#else
                                                                 nullptr,
                                                                 nullptr,
#endif
                                                                         Xrot,
                                                                         cols_local,
                                                                         knn_ids_flat,
                                                                         knn_dists_flat,
                                                                         knn_real,
                                                                         knn_umap,
                                                                         cid,
                                                                         k_virtual,
                                                                         umap_reencode_cfg.ils_iters,
                                                                         umap_reencode_cfg.icm_iters,
                                                                         umap_reencode_cfg.perturb_k,
                                                                         seed,
                                                                         &X_virt,
                                                                         &B_virt,
                                                                         &a_virt,
                                                                         &local_err)) {
                        fail(local_err);
                        continue;
                    }
                    if (stlq::linkage::LinkageBuildProfileEnabled()) {
                        t_time.prof.bad_umap_s += stlq::linkage::LinkageBuildWallNowS() - t0_bad_umap;
                    }
                    if (profile_timing) {
                        t_time.bad_knn_umap += now_s() - t0_bad;
                    }
                    const double t0_linkage = profile_timing ? now_s() : 0.0;
                    const double t0_linkage_wall = stlq::linkage::LinkageBuildProfileEnabled()
                                                       ? stlq::linkage::LinkageBuildWallNowS()
                                                       : 0.0;
                    // Release dist table now.
                    knn_dists_flat.clear();

                    n_virt = B_virt.cols;

                    // Prepare real reconstruction buffer.
                    ColMajorMatrix<float>& R_full_shared = buf.R_full_shared;
                    ReconstructAllIntoFixedRoot(train.C_root.books, cid, B_full, a, &R_full_shared);
                    ColMajorMatrix<float>& R_virt = buf.R_virt;
                    R_virt.rows = d;
                    R_virt.cols = n_virt;
                    const std::size_t need_virt =
                        static_cast<std::size_t>(d) * static_cast<std::size_t>(std::max(0, n_virt));
                    if (R_virt.data.size() < need_virt) {
                        R_virt.data.resize(need_virt);
                    }
                    if (need_virt > 0) {
                        std::fill_n(R_virt.data.data(), need_virt, 0.0f);
                    }

                    LinkageStructure& linkage_bad = buf.linkage_bad;
                    try {
                        if (linkage_cfg.use_ils && linkage_cfg.ils_rounds > 0 && linkage_cfg.ils_perturb_layers > 0) {
                            LinkageMultiCenterTwoCodebookImplCore<true>(linkage_cfg,
                                                                        cid,
                                                                        cfg.hnsw,
#if defined(STLQ_ENABLE_CUDA)
                                                                        &cfg.runtime, cuda_pool_ptr,
#else
                                                           nullptr,
#endif
                                                                        Xrot,
                                                                        cols_local,
                                                                        n_virt > 0 ? B_virt.data.data() : nullptr,
                                                                        n_virt > 0 ? a_virt.data.data() : nullptr,
                                                                        n_virt,
                                                                        train.C_root,
                                                                        pre_root,
                                                                        train.C_one,
                                                                        pre_one,
                                                                        G_one_root,
                                                                        knn_ids_flat,
                                                                        knn_real,
                                                                        &B_full,
                                                                        &a,
                                                                        &R_full_shared,
                                                                        &R_virt,
                                                                        &linkage_bad);
                        }
                        else {
                            LinkageMultiCenterTwoCodebookImplCore<false>(linkage_cfg,
                                                                         cid,
                                                                         cfg.hnsw,
#if defined(STLQ_ENABLE_CUDA)
                                                                         &cfg.runtime, cuda_pool_ptr,
#else
                                                               nullptr,
#endif
                                                                         Xrot,
                                                                         cols_local,
                                                                         n_virt > 0 ? B_virt.data.data() : nullptr,
                                                                         n_virt > 0 ? a_virt.data.data() : nullptr,
                                                                         n_virt,
                                                                         train.C_root,
                                                                         pre_root,
                                                                         train.C_one,
                                                                         pre_one,
                                                                         G_one_root,
                                                                         knn_ids_flat,
                                                                         knn_real,
                                                                         &B_full,
                                                                         &a,
                                                                         &R_full_shared,
                                                                         &R_virt,
                                                                         &linkage_bad);
                        }
                    }
                    catch (const std::exception& e) {
                        fail(std::string("Bad cluster linkage failed: ") + e.what());
                        continue;
                    }
                    // One cluster result.
                    cluster_out = linkage_bad.clusters.front();
                    ls_failures = 0;

                    // `R_full_shared` is updated in-place by the multi-center core to hold linkaged recon for real nodes.
                    if (n_real > 0) {
                        const MseSumStats mse = MseSumStatsFromReconLocal(Xrot, R_full_shared);
                        cluster_mse_sum_local = mse.sum;
                        if (std::isfinite(mse.min)) {
                            cluster_linkage_mse.mean = static_cast<float>(mse.sum / static_cast<double>(n_real));
                            cluster_linkage_mse.min = static_cast<float>(mse.min);
                            cluster_linkage_mse.max = static_cast<float>(mse.max);
                        }
                    }
                    if (profile_timing) {
                        t_time.linkage_core += now_s() - t0_linkage;
                    }
                    if (stlq::linkage::LinkageBuildProfileEnabled()) {
                        t_time.prof.linkage_bad_s += stlq::linkage::LinkageBuildWallNowS() - t0_linkage_wall;
                    }
                }
                else {
                    // Good cluster or virtual disabled: standard inner-to-outer linkageing.
                    const double t0_linkage = profile_timing ? now_s() : 0.0;
                    const double t0_linkage_wall = stlq::linkage::LinkageBuildProfileEnabled()
                                                       ? stlq::linkage::LinkageBuildWallNowS()
                                                       : 0.0;
                    if (linkage_cfg.use_ils && linkage_cfg.ils_rounds > 0 && linkage_cfg.ils_perturb_layers > 0) {
                        LinkageTwoCodebookInnerToOuterOneCluster<true>(linkage_cfg, cfg.hnsw,
#if defined(STLQ_ENABLE_CUDA)
                                                                       &cfg.runtime, cuda_pool_ptr,
#else
                                                            nullptr,
#endif
                                                                       cid,
                                                                       Xrot,
                                                                       train.C_root, pre_root,
                                                                       train.C_one, pre_one,
                                                                       G_one_root,
                                                                       &B_full, &a,
                                                                       &cluster_out,
                                                                       &ls_failures,
                                                                       &cluster_linkage_mse,
                                                                       &cluster_mse_sum_local);
                    }
                    else {
                        LinkageTwoCodebookInnerToOuterOneCluster<false>(linkage_cfg, cfg.hnsw,
#if defined(STLQ_ENABLE_CUDA)
                                                                        &cfg.runtime, cuda_pool_ptr,
#else
                                                             nullptr,
#endif
                                                                        cid,
                                                                        Xrot,
                                                                        train.C_root, pre_root,
                                                                        train.C_one, pre_one,
                                                                        G_one_root,
                                                                        &B_full, &a,
                                                                        &cluster_out,
                                                                        &ls_failures,
                                                                        &cluster_linkage_mse,
                                                                        &cluster_mse_sum_local);
                    }
                    if (profile_timing) {
                        t_time.linkage_core += now_s() - t0_linkage;
                    }
                    if (stlq::linkage::LinkageBuildProfileEnabled()) {
                        t_time.prof.linkage_good_s += stlq::linkage::LinkageBuildWallNowS() - t0_linkage_wall;
                    }
                }

                // Per-cluster stats (cid-indexed, deterministic reduction later).
                if (cluster_out.n_real > 0) {
                    cluster_real[static_cast<std::size_t>(cid)] = static_cast<std::uint64_t>(cluster_out.n_real);
                    std::uint64_t linkaged = 0;
                    const int n_virt_local = std::max(0, cluster_out.n_virtual);
                    for (int i = 0; i < cluster_out.n_real; ++i) {
                        const int real_local = n_virt_local + i;
                        if (cluster_out.parent_local[static_cast<std::size_t>(real_local)] != 0) {
                            linkaged += 1;
                        }
                    }
                    cluster_linkaged[static_cast<std::size_t>(cid)] = linkaged;

                    const int actual_max_depth =
                        (cluster_out.depth_offsets.size() >= 2)
                            ? static_cast<int>(cluster_out.depth_offsets.size()) - 2
                            : 0;
                    cluster_max_depth[static_cast<std::size_t>(cid)] = actual_max_depth;

                    double depth_sum_local = 0.0;
                    for (int dep = 0; dep <= actual_max_depth; ++dep) {
                        const int lo = cluster_out.depth_offsets[static_cast<std::size_t>(dep)];
                        const int hi = cluster_out.depth_offsets[static_cast<std::size_t>(dep + 1)];
                        const int cnt = std::max(0, hi - lo);
                        depth_sum_local += static_cast<double>(dep) * static_cast<double>(cnt);
                    }
                    cluster_depth_sum[static_cast<std::size_t>(cid)] = depth_sum_local;

                    cluster_mse_sum[static_cast<std::size_t>(cid)] = cluster_mse_sum_local;
                    cluster_mse_min[static_cast<std::size_t>(cid)] = static_cast<double>(cluster_linkage_mse.min);
                    cluster_mse_max[static_cast<std::size_t>(cid)] = static_cast<double>(cluster_linkage_mse.max);
                }

                // Prepare depth-ordered outputs for linkage_list (real ids / parent / depth_offsets / codes_small / coeffs_small).
                const int n_real_out = cluster_out.n_real;
                const int n_virt_out = cluster_out.n_virtual;
                std::vector<std::uint32_t>& real_ids_depth_order = buf.real_ids_depth_order;
                std::vector<std::uint32_t>& parent_u32 = buf.parent_u32;
                std::vector<std::uint32_t>& depth_offsets_u32 = buf.depth_offsets_u32;
                std::vector<std::uint8_t>& codes_small_depth = buf.codes_small_depth;
                std::vector<float>& coeffs_small_depth = buf.coeffs_small_depth;
                std::vector<std::uint8_t>& code0_one_depth = buf.code0_one_depth;
                std::vector<float>& a0_depth = buf.a0_depth;
                std::vector<std::uint8_t>& virt_codes_small = buf.virt_codes_small;
                std::vector<float>& virt_coeffs_small = buf.virt_coeffs_small;
                std::vector<float>& virt_a0 = buf.virt_a0;

                real_ids_depth_order.resize(static_cast<std::size_t>(n_real_out));
                parent_u32.resize(static_cast<std::size_t>(n_real_out));
                depth_offsets_u32.assign(static_cast<std::size_t>(fixed_depth_len),
                                         static_cast<std::uint32_t>(n_real_out));
                if (cluster_out.depth_offsets.size() > static_cast<std::size_t>(fixed_depth_len)) {
                    fail("BuildLinkageTwoCodebookVirtualStreamingByCluster: depth_offsets overflow.");
                    continue;
                }
                for (std::size_t i = 0; i < cluster_out.depth_offsets.size(); ++i) {
                    depth_offsets_u32[i] = static_cast<std::uint32_t>(cluster_out.depth_offsets[i]);
                }

                const auto n_real_out_sz = static_cast<std::size_t>(n_real_out);
                const std::size_t codes_real_sz =
                    static_cast<std::size_t>(m_codes) * static_cast<std::size_t>(n_real_out);
                const auto code0_bytes =
                    static_cast<std::size_t>(out_cfg.code0_width_bytes);
                const std::size_t code0_real_sz = n_real_out_sz * code0_bytes;

                codes_small_depth.resize(codes_real_sz);
                coeffs_small_depth.resize(codes_real_sz);
                code0_one_depth.resize(code0_real_sz);
                a0_depth.resize(n_real_out_sz);
                const int real_base = n_virt_out;
                for (int pos = 0; pos < n_real_out; ++pos) {
                    const auto p =
                        static_cast<std::uint32_t>(cluster_out.parent_local[static_cast<std::size_t>(real_base + pos)]);
                    parent_u32[static_cast<std::size_t>(pos)] = p;
                    const int old_local = cluster_out.indices[static_cast<std::size_t>(pos)];
                    // NOTE: `old_local` originates from `cluster_out.indices` which is constructed from
                    // cluster membership; it is expected to be within [0, n_real).
                    real_ids_depth_order[static_cast<std::size_t>(pos)] =
                        ids[static_cast<std::size_t>(old_local)];
                    a0_depth[static_cast<std::size_t>(pos)] = a(0, old_local);

                    // depth>0: layer-0 uses C_one (persist code0); depth=0 stores 0.
                    const std::size_t off0 = static_cast<std::size_t>(pos) * code0_bytes;
                    if (p != 0) {
                        const auto code0_one = static_cast<std::uint16_t>(B_full(0, old_local));
                        if (code0_bytes == 1) {
                            code0_one_depth[off0] = static_cast<std::uint8_t>(code0_one);
                        }
                        else {
                            std::memcpy(code0_one_depth.data() + off0, &code0_one, sizeof(std::uint16_t));
                        }
                    }
                    else {
                        if (code0_bytes == 1) {
                            code0_one_depth[off0] = 0;
                        }
                        else {
                            const std::uint16_t z = 0;
                            std::memcpy(code0_one_depth.data() + off0, &z, sizeof(std::uint16_t));
                        }
                    }

                    for (int l = 1; l < m; ++l) {
                        const std::size_t offc =
                            static_cast<std::size_t>(pos) * static_cast<std::size_t>(m_codes) +
                            static_cast<std::size_t>(l - 1);
                        codes_small_depth[offc] = B_full(l, old_local);
                        coeffs_small_depth[offc] = a(l, old_local);
                    }
                }

                if (n_virt_out > 0) {
                    const auto n_virt_out_sz = static_cast<std::size_t>(n_virt_out);
                    const std::size_t codes_virt_sz =
                        static_cast<std::size_t>(m_codes) * static_cast<std::size_t>(n_virt_out);
                    virt_codes_small.resize(codes_virt_sz);
                    virt_coeffs_small.resize(codes_virt_sz);
                    virt_a0.resize(n_virt_out_sz);
                    for (int v = 0; v < n_virt_out; ++v) {
                        virt_a0[static_cast<std::size_t>(v)] = a_virt(0, v);
                        for (int l = 1; l < m; ++l) {
                            const std::size_t offc =
                                static_cast<std::size_t>(v) * static_cast<std::size_t>(m_codes) +
                                static_cast<std::size_t>(l - 1);
                            virt_codes_small[offc] = B_virt(l, v);
                            virt_coeffs_small[offc] = a_virt(l, v);
                        }
                    }
                }
                else {
                    virt_codes_small.resize(0);
                    virt_coeffs_small.resize(0);
                    virt_a0.resize(0);
                }

                if (use_coeff_codec) {
                    // Encode coefficients as int8 (lossy) + Huffman (lossless), grouped as:
                    // - root: depth==0 nodes (real parent==0 + all virtual roots)
                    // - linkage: depth>0 residual nodes (real parent!=0)
                    //
                    // NOTE: we store only Huffman payloads + lens + per-layer scales; no float coeffs are persisted in linkage_list.
                    const int nc = n_real_out + n_virt_out;
                    if (nc <= 0) {
                        // Empty cluster: still write deterministic empty codec record.
                        buf.lens_root = coeff_lens_zeros;
                        buf.lens_linkage = coeff_lens_zeros;
                        buf.payload_root = coeff_payload_empty;
                        buf.payload_linkage = coeff_payload_empty;
                        if (!coeff_writer.WriteClusterAt(cid,
                                                         coeff_scales_ones,
                                                         coeff_scales_ones,
                                                         buf.lens_root,
                                                         buf.lens_linkage,
                                                         buf.payload_root,
                                                         buf.payload_linkage,
                                                         &local_err)) {
                            fail(local_err);
                            continue;
                        }
                    }
                    else {
                        buf.a_layer_major.resize(static_cast<std::size_t>(m) * static_cast<std::size_t>(nc));
                        buf.is_linkage.resize(static_cast<std::size_t>(nc));
                        buf.q_tmp.resize(static_cast<std::size_t>(m) * static_cast<std::size_t>(nc));

                        for (int v = 0; v < n_virt_out; ++v) {
                            buf.is_linkage[static_cast<std::size_t>(v)] = 0u;
                            buf.a_layer_major[static_cast<std::size_t>(v)] = virt_a0[static_cast<std::size_t>(v)];
                            for (int l = 1; l < m; ++l) {
                                const std::size_t offc =
                                    static_cast<std::size_t>(v) * static_cast<std::size_t>(m_codes) +
                                    static_cast<std::size_t>(l - 1);
                                buf.a_layer_major[static_cast<std::size_t>(l) * static_cast<std::size_t>(nc) +
                                        static_cast<std::size_t>(v)] =
                                    virt_coeffs_small[offc];
                            }
                        }
                        for (int pos = 0; pos < n_real_out; ++pos) {
                            const int local = real_base + pos;
                            buf.is_linkage[static_cast<std::size_t>(local)] =
                                (parent_u32[static_cast<std::size_t>(pos)] != 0) ? 1u : 0u;
                            buf.a_layer_major[static_cast<std::size_t>(local)] = a0_depth[static_cast<std::size_t>(
                                pos)];
                            for (int l = 1; l < m; ++l) {
                                const std::size_t offc =
                                    static_cast<std::size_t>(pos) * static_cast<std::size_t>(m_codes) +
                                    static_cast<std::size_t>(l - 1);
                                buf.a_layer_major[static_cast<std::size_t>(l) * static_cast<std::size_t>(nc) +
                                        static_cast<std::size_t>(local)] =
                                    coeffs_small_depth[offc];
                            }
                        }

                        CoeffQuantConfig qcfg;
                        qcfg.bits_per_layer = cfg.large.linkage_coeff_codec.bits_per_layer;
                        qcfg.p_first_candidates = cfg.large.linkage_coeff_codec.p_first_candidates;
                        qcfg.p_rest_candidates = cfg.large.linkage_coeff_codec.p_rest_candidates;
                        qcfg.use_weighted_quantile = cfg.large.linkage_coeff_codec.use_weighted_quantile;
                        qcfg.allow_clip = cfg.large.linkage_coeff_codec.allow_clip;
                        qcfg.fit_scale = cfg.large.linkage_coeff_codec.fit_scale;

                        HuffmanConfig hcfg;
                        hcfg.alphabet_size = 0;
                        if (cfg.large.linkage_coeff_codec.granularity == "layer" ||
                            cfg.large.linkage_coeff_codec.granularity == "per_layer") {
                            hcfg.granularity = HuffmanGranularity::kPerLayer;
                        }
                        else {
                            hcfg.granularity = HuffmanGranularity::kPerClusterAllLayers;
                        }

                        // Build 1-based code indices for weighted quantile and recon-aligned scale fitting.
                        buf.B_code1_layer_major.resize(static_cast<std::size_t>(m) * static_cast<std::size_t>(nc));
                        auto read_code0_one = [&](int pos) -> int
                        {
                            const std::size_t i = static_cast<std::size_t>(pos);
                            if (out_cfg.code0_width_bytes == 1) {
                                return static_cast<int>(code0_one_depth[i]);
                            }
                            if (out_cfg.code0_width_bytes == 2) {
                                std::uint16_t v16 = 0;
                                std::memcpy(&v16, code0_one_depth.data() + i * sizeof(std::uint16_t),
                                            sizeof(std::uint16_t));
                                return static_cast<int>(v16);
                            }
                            std::uint32_t v32 = 0;
                            std::memcpy(&v32, code0_one_depth.data() + i * sizeof(std::uint32_t),
                                        sizeof(std::uint32_t));
                            return static_cast<int>(v32);
                        };
                        // layer0 codes (root vs one).
                        for (int pos = 0; pos < n_real_out; ++pos) {
                            const int local = real_base + pos;
                            const bool ilinkage = (buf.is_linkage[static_cast<std::size_t>(local)] != 0);
                            const int code0 = ilinkage ? read_code0_one(pos) : cid;
                            buf.B_code1_layer_major[static_cast<std::size_t>(local)] = static_cast<std::uint32_t>(code0
                                + 1);
                        }
                        for (int v = 0; v < n_virt_out; ++v) {
                            buf.B_code1_layer_major[static_cast<std::size_t>(v)] = static_cast<std::uint32_t>(cid + 1);
                        }
                        // layers 1..m-1 codes.
                        for (int l = 1; l < m; ++l) {
                            for (int pos = 0; pos < n_real_out; ++pos) {
                                const int local = real_base + pos;
                                const std::size_t offc =
                                    static_cast<std::size_t>(pos) * static_cast<std::size_t>(m_codes) +
                                    static_cast<std::size_t>(l - 1);
                                const int code = static_cast<int>(codes_small_depth[offc]);
                                buf.B_code1_layer_major[static_cast<std::size_t>(l) * static_cast<std::size_t>(nc) +
                                        static_cast<std::size_t>(local)] =
                                    static_cast<std::uint32_t>(code + 1);
                            }
                            for (int v = 0; v < n_virt_out; ++v) {
                                const std::size_t offc =
                                    static_cast<std::size_t>(v) * static_cast<std::size_t>(m_codes) +
                                    static_cast<std::size_t>(l - 1);
                                const int code = static_cast<int>(virt_codes_small[offc]);
                                buf.B_code1_layer_major[static_cast<std::size_t>(l) * static_cast<std::size_t>(nc) +
                                        static_cast<std::size_t>(v)] =
                                    static_cast<std::uint32_t>(code + 1);
                            }
                        }

                        // Stage-1: per-layer quantization (optionally weighted) to produce q (int8).
                        CoeffQuantGroupResult qres_root;
                        CoeffQuantGroupResult qres_one;
                        std::fill(buf.q_tmp.begin(), buf.q_tmp.end(), std::int8_t(0));
                        try {
                            const std::vector<Span<const float>> empty_norms;
                            const auto& n2_root = qcfg.use_weighted_quantile
                                                      ? coeff_pre->norm2_root_spans
                                                      : empty_norms;
                            const auto& n2_one = qcfg.use_weighted_quantile ? coeff_pre->norm2_one_spans : empty_norms;
                            QuantizeClusterCoeffsSplitRootLinkage(
                                m,
                                nc,
                                Span<const float>(buf.a_layer_major.data(), buf.a_layer_major.size()),
                                Span<const std::uint32_t>(buf.B_code1_layer_major.data(),
                                                          buf.B_code1_layer_major.size()),
                                Span<const std::uint8_t>(buf.is_linkage.data(), buf.is_linkage.size()),
                                n2_root,
                                n2_one,
                                qcfg,
                                Span<std::int8_t>(buf.q_tmp.data(), buf.q_tmp.size()),
                                &qres_root,
                                &qres_one);
                        }
                        catch (const std::exception& e) {
                            fail(std::string(
                                    "BuildLinkageTwoCodebookVirtualStreamingByCluster: coeff quantize failed: ") + e.
                                what());
                            continue;
                        }

                        std::vector<float> scale_root = qres_root.scales;
                        std::vector<float> scale_one = qres_one.scales;

                        // Stage-2/3/4: recon-aligned global scale fit + optional ICM refine of q.
                        const int sweeps = std::max(0, cfg.large.linkage_coeff_codec.q_refine_sweeps);
                        const int Lref = (cfg.large.linkage_coeff_codec.q_refine_max_layer <= 0)
                                             ? m
                                             : std::min(m, cfg.large.linkage_coeff_codec.q_refine_max_layer);
                        const int step_limit = std::max(0, cfg.large.linkage_coeff_codec.q_refine_step_limit);

                        std::vector<int> Qvec(static_cast<std::size_t>(m), 0);
                        for (int l = 0; l < m; ++l) {
                            const int bits = std::max(2, qcfg.bits_per_layer[static_cast<std::size_t>(l)]);
                            Qvec[static_cast<std::size_t>(l)] = (1 << (bits - 1)) - 1;
                        }

                        // Compute g0s for this cid (dot between C_root[0][:,cid] and flattened C_root small layers).
                        buf.g0s_root_small.resize(static_cast<std::size_t>(coeff_pre->H_root_small));
                        const float* c0 = train.C_root.books[0].Col(cid);
                        for (int j = 0; j < coeff_pre->H_root_small; ++j) {
                            const float* cs = coeff_pre->C_root_small.Col(j);
                            float s = 0.0f;
#pragma omp simd reduction(+:s)
                            for (int r = 0; r < d; ++r) {
                                s += c0[r] * cs[r];
                            }
                            buf.g0s_root_small[static_cast<std::size_t>(j)] = s;
                        }

                        auto dot_root = [&](int l, int code_l0, int k, int code_k0) -> float
                        {
                            if (l == 0 && k == 0) {
                                return coeff_pre->norm0_root[static_cast<std::size_t>(cid)];
                            }
                            if (l == 0 && k > 0) {
                                const int fk = coeff_pre->root_small_offsets[static_cast<std::size_t>(k)] + code_k0;
                                return buf.g0s_root_small[static_cast<std::size_t>(fk)];
                            }
                            if (l > 0 && k == 0) {
                                const int fl = coeff_pre->root_small_offsets[static_cast<std::size_t>(l)] + code_l0;
                                return buf.g0s_root_small[static_cast<std::size_t>(fl)];
                            }
                            const int fl = coeff_pre->root_small_offsets[static_cast<std::size_t>(l)] + code_l0;
                            const int fk = coeff_pre->root_small_offsets[static_cast<std::size_t>(k)] + code_k0;
                            return coeff_pre->G_root_small(fl, fk);
                        };
                        auto dot_one = [&](int l, int code_l0, int k, int code_k0) -> float
                        {
                            const int fl = coeff_pre->one_offsets[static_cast<std::size_t>(l)] + code_l0;
                            const int fk = coeff_pre->one_offsets[static_cast<std::size_t>(k)] + code_k0;
                            return coeff_pre->G_one_all(fl, fk);
                        };

                        auto fit_scales = [&](const std::vector<int>& idx,
                                              const std::vector<float>& fallback,
                                              auto&& dot_fn) -> std::vector<float>
                        {
                            std::vector<double> A(static_cast<std::size_t>(m) * static_cast<std::size_t>(m), 0.0);
                            std::vector<double> b(static_cast<std::size_t>(m), 0.0);
                            std::vector<double> x(static_cast<std::size_t>(m), 0.0);
                            std::vector<double> dot_t(static_cast<std::size_t>(m), 0.0);

                            for (int ii : idx) {
                                // codes (0-based) and q/coeff.
                                int code[32];
                                int qv[32];
                                double av[32];
                                for (int l = 0; l < m; ++l) {
                                    code[l] = static_cast<int>(buf.B_code1_layer_major[static_cast<std::size_t>(l) *
                                        static_cast<std::size_t>(nc) +
                                        static_cast<std::size_t>(ii)]) - 1;
                                    qv[l] = static_cast<int>(buf.q_tmp[static_cast<std::size_t>(l) * static_cast<
                                            std::size_t>(nc) +
                                        static_cast<std::size_t>(ii)]);
                                    av[l] = static_cast<double>(buf.a_layer_major[static_cast<std::size_t>(l) *
                                        static_cast<std::size_t>(nc) +
                                        static_cast<std::size_t>(ii)]);
                                }
                                // dot_t[l] = <c_l, t>
                                for (int l = 0; l < m; ++l) {
                                    double acc = 0.0;
                                    for (int k = 0; k < m; ++k) {
                                        acc += av[k] * static_cast<double>(dot_fn(l, code[l], k, code[k]));
                                    }
                                    dot_t[static_cast<std::size_t>(l)] = acc;
                                }
                                for (int l = 0; l < m; ++l) {
                                    const int ql = qv[l];
                                    if (ql == 0) continue;
                                    b[static_cast<std::size_t>(l)] += static_cast<double>(ql) * dot_t[static_cast<
                                        std::size_t>(l)];
                                    for (int k = 0; k < m; ++k) {
                                        const int qk = qv[k];
                                        if (qk == 0) continue;
                                        A[static_cast<std::size_t>(l) * static_cast<std::size_t>(m) + static_cast<
                                                std::size_t>(k)] +=
                                            static_cast<double>(ql) * static_cast<double>(qk) *
                                            static_cast<double>(dot_fn(l, code[l], k, code[k]));
                                    }
                                }
                            }
                            if (!SolveSymPosdefCholeskyRetry(m, A.data(), b.data(), x.data())) {
                                return fallback;
                            }
                            std::vector<float> out(static_cast<std::size_t>(m), 0.0f);
                            for (int l = 0; l < m; ++l) {
                                out[static_cast<std::size_t>(l)] = static_cast<float>(x[static_cast<std::size_t>(l)]);
                            }
                            return out;
                        };

                        auto refine_q_icm = [&](const std::vector<int>& idx,
                                                const std::vector<float>& scales,
                                                auto&& dot_fn)
                        {
                            if (idx.empty() || sweeps <= 0) {
                                return;
                            }
                            std::vector<double> dot_t(static_cast<std::size_t>(m), 0.0);
                            for (int ii : idx) {
                                int code[32];
                                int qv[32];
                                double av[32];
                                for (int l = 0; l < m; ++l) {
                                    code[l] = static_cast<int>(buf.B_code1_layer_major[static_cast<std::size_t>(l) *
                                        static_cast<std::size_t>(nc) +
                                        static_cast<std::size_t>(ii)]) - 1;
                                    qv[l] = static_cast<int>(buf.q_tmp[static_cast<std::size_t>(l) * static_cast<
                                            std::size_t>(nc) +
                                        static_cast<std::size_t>(ii)]);
                                    av[l] = static_cast<double>(buf.a_layer_major[static_cast<std::size_t>(l) *
                                        static_cast<std::size_t>(nc) +
                                        static_cast<std::size_t>(ii)]);
                                }
                                for (int l = 0; l < m; ++l) {
                                    double acc = 0.0;
                                    for (int k = 0; k < m; ++k) {
                                        acc += av[k] * static_cast<double>(dot_fn(l, code[l], k, code[k]));
                                    }
                                    dot_t[static_cast<std::size_t>(l)] = acc;
                                }
                                for (int sw = 0; sw < sweeps; ++sw) {
                                    for (int l = 0; l < Lref; ++l) {
                                        const auto sl = static_cast<double>(scales[static_cast<std::size_t>(l)]);
                                        if (sl == 0.0) {
                                            qv[l] = 0;
                                            continue;
                                        }
                                        const auto cc = static_cast<double>(dot_fn(l, code[l], l, code[l]));
                                        if (cc == 0.0) {
                                            qv[l] = 0;
                                            continue;
                                        }
                                        double sum_other = 0.0;
                                        for (int k = 0; k < m; ++k) {
                                            if (k == l) continue;
                                            const int qk = qv[k];
                                            if (qk == 0) continue;
                                            sum_other += static_cast<double>(scales[static_cast<std::size_t>(k)]) *
                                                static_cast<double>(qk) *
                                                static_cast<double>(dot_fn(l, code[l], k, code[k]));
                                        }
                                        const double beta = sl * (dot_t[static_cast<std::size_t>(l)] - sum_other);
                                        const double alpha = (sl * sl) * cc;
                                        if (alpha == 0.0) {
                                            continue;
                                        }
                                        const int Q = Qvec[static_cast<std::size_t>(l)];
                                        const int qcur = qv[l];
                                        int qnew = qcur;
                                        if (step_limit > 0) {
                                            const int lo = std::max(-Q, qcur - step_limit);
                                            const int hi = std::min(Q, qcur + step_limit);
                                            double bestv = alpha * double(qcur) * double(qcur) - 2.0 * beta * double(
                                                qcur);
                                            for (int cand = lo; cand <= hi; ++cand) {
                                                const double v = alpha * double(cand) * double(cand) - 2.0 * beta *
                                                    double(cand);
                                                if (v < bestv) {
                                                    bestv = v;
                                                    qnew = cand;
                                                }
                                            }
                                        }
                                        else {
                                            const double qcont = beta / alpha;
                                            qnew = static_cast<int>(std::llround(qcont));
                                            qnew = std::min(Q, std::max(-Q, qnew));
                                        }
                                        qv[l] = qnew;
                                    }
                                }
                                for (int l = 0; l < Lref; ++l) {
                                    const int q = std::min(127, std::max(-128, qv[l]));
                                    buf.q_tmp[static_cast<std::size_t>(l) * static_cast<std::size_t>(nc) +
                                        static_cast<std::size_t>(ii)] = static_cast<std::int8_t>(q);
                                }
                            }
                        };

                        // Split indices for groups.
                        std::vector<int> idx_root;
                        std::vector<int> idx_linkage;
                        idx_root.reserve(static_cast<std::size_t>(nc));
                        idx_linkage.reserve(static_cast<std::size_t>(nc));
                        for (int ii = 0; ii < nc; ++ii) {
                            if (buf.is_linkage[static_cast<std::size_t>(ii)] == 0) {
                                idx_root.push_back(ii);
                            }
                            else {
                                idx_linkage.push_back(ii);
                            }
                        }

                        if (qcfg.fit_scale) {
                            scale_root = fit_scales(idx_root, scale_root, dot_root);
                            scale_one = fit_scales(idx_linkage, scale_one, dot_one);
                        }
                        if (sweeps > 0) {
                            refine_q_icm(idx_root, scale_root, dot_root);
                            refine_q_icm(idx_linkage, scale_one, dot_one);
                            if (qcfg.fit_scale) {
                                scale_root = fit_scales(idx_root, scale_root, dot_root);
                                scale_one = fit_scales(idx_linkage, scale_one, dot_one);
                            }
                        }

                        ClusterCoeffCompressed comp;
                        try {
                            comp = CompressClusterCoeffsFromQ(
                                static_cast<std::uint32_t>(cid),
                                m,
                                nc,
                                Span<const std::int8_t>(buf.q_tmp.data(), buf.q_tmp.size()),
                                Span<const std::uint8_t>(buf.is_linkage.data(), buf.is_linkage.size()),
                                scale_root,
                                scale_one,
                                qcfg,
                                hcfg);
                        }
                        catch (const std::exception& e) {
                            fail(std::string("BuildLinkageTwoCodebookVirtualStreamingByCluster: coeff codec failed: ") +
                                e.what());
                            continue;
                        }

                        const int streams = coeff_meta ? coeff_meta->streams : 0;
                        const int S = coeff_meta ? coeff_meta->S : 0;
                        buf.lens_root.assign(static_cast<std::size_t>(streams),
                                             std::vector<std::uint8_t>(static_cast<std::size_t>(S), 0));
                        buf.lens_linkage.assign(static_cast<std::size_t>(streams),
                                                std::vector<std::uint8_t>(static_cast<std::size_t>(S), 0));
                        buf.payload_root.assign(static_cast<std::size_t>(streams), {});
                        buf.payload_linkage.assign(static_cast<std::size_t>(streams), {});

                        auto fill_group = [&](const ClusterGroupCompressed& g,
                                              std::vector<std::vector<std::uint8_t>>* lens_out,
                                              std::vector<std::vector<std::uint8_t>>* payload_out)
                        {
                            if (g.models.empty() || g.payloads.empty()) {
                                return;
                            }
                            if (static_cast<int>(g.models.size()) == 1) {
                                (*lens_out)[0] = g.models[0].len;
                                (*payload_out)[0] = g.payloads[0];
                                return;
                            }
                            for (int l = 0; l < std::min(streams, static_cast<int>(g.models.size())); ++l) {
                                (*lens_out)[static_cast<std::size_t>(l)] = g.models[static_cast<std::size_t>(l)].len;
                                (*payload_out)[static_cast<std::size_t>(l)] = g.payloads[static_cast<std::size_t>(l)];
                            }
                        };
                        fill_group(comp.root, &buf.lens_root, &buf.payload_root);
                        fill_group(comp.linkage, &buf.lens_linkage, &buf.payload_linkage);

                        if (!coeff_writer.WriteClusterAt(cid,
                                                         comp.scale_root,
                                                         comp.scale_linkage,
                                                         buf.lens_root,
                                                         buf.lens_linkage,
                                                         buf.payload_root,
                                                         buf.payload_linkage,
                                                         &local_err)) {
                            fail(local_err);
                            continue;
                        }
                    }
                }

                const double t0_write = profile_timing ? now_s() : 0.0;
                if (!random_writer.WriteClusterAt(cid, plan,
                                                  real_ids_depth_order,
                                                  parent_u32,
                                                  depth_offsets_u32,
                                                  codes_small_depth,
                                                  coeffs_small_depth,
                                                  code0_one_depth,
                                                  a0_depth,
                                                  virt_codes_small,
                                                  virt_coeffs_small,
                                                  virt_a0,
                                                  &local_err)) {
                    fail(local_err);
                    continue;
                }
                if (profile_timing) {
                    t_time.write += now_s() - t0_write;
                }

                if (use_linkage_ckpt) {
                    ClusterCkptStatsV1 s;
                    const auto i = static_cast<std::size_t>(cid);
                    s.real = cluster_real[i];
                    s.linkaged = cluster_linkaged[i];
                    s.depth_sum = cluster_depth_sum[i];
                    s.max_depth = static_cast<std::int32_t>(cluster_max_depth[i]);
                    s.mse_sum = cluster_mse_sum[i];
                    s.mse_min = cluster_mse_min[i];
                    s.mse_max = cluster_mse_max[i];
                    std::string ckpt_err;
                    if (!ckpt.MarkDone(cid, s, &ckpt_err)) {
                        fail("BuildLinkageTwoCodebookVirtualStreamingByCluster: checkpoint MarkDone failed: " +
                            ckpt_err);
                        continue;
                    }
                }

                if (ls_failures > 0) {
                    LogWarn("Cluster " + std::to_string(cid) + " LS failures: " + std::to_string(ls_failures));
                }

                bump_progress();
            }
        } // omp parallel

        if (async_io) {
            async_io->Stop();
            if (profile_timing && !timing_tls.empty()) {
                timing_tls[0].read_ivf += async_io->read_ivf_s();
                timing_tls[0].read_raw += async_io->read_raw_s();
                timing_tls[0].read_base += async_io->read_base_s();
            }
        }

        if (profile_timing) {
            const double t1 = wall_now_s();
            t_wall_pass_b = t1 - t_wall_t0;
            t_wall_t0 = t1;
        }

        // Ensure the progress line is terminated even if thread-0 didn't observe the final increment.
        // (Progress prints are on stderr; LogInfo uses stdout and may otherwise interleave on the same line.)
        {
            const int final_done = done.load(std::memory_order_relaxed);
            std::fprintf(stderr, "\rLinkage build: %d/%d clusters   \n", final_done, nlist);
            std::fflush(stderr);
        }

        if (!ok.load()) {
            if (err) {
                *err = first_err.empty() ? "BuildLinkageTwoCodebookVirtualStreamingByCluster: failed." : first_err;
            }
            return false;
        }
        if (!random_writer.Finish(err)) {
            return false;
        }

        if (profile_timing) {
            const double t1 = wall_now_s();
            t_wall_writer_finish = t1 - t_wall_t0;
            t_wall_t0 = t1;
        }

        if (use_coeff_codec) {
            if (!coeff_writer.Finish(err)) {
                return false;
            }
            // Refresh stats after Finish() sets payload_bytes.
            coeff_stats = coeff_writer.stats();
        }

        if (profile_timing) {
            const double t1 = wall_now_s();
            t_wall_coeff_finish = t1 - t_wall_t0;
            t_wall_t0 = t1;
        }

        // Deterministic (cid-ordered) reduction for summary stats.
        std::uint64_t total_real = 0;
        std::uint64_t total_linkaged = 0;
        double depth_sum = 0.0;
        int max_depth = 0;

        double mse_sum = 0.0;
        double mse_min = std::numeric_limits<double>::infinity();
        double mse_max = 0.0;

        for (int cid = 0; cid < nlist; ++cid) {
            const auto i = static_cast<std::size_t>(cid);
            total_real += cluster_real[i];
            total_linkaged += cluster_linkaged[i];
            depth_sum += cluster_depth_sum[i];
            max_depth = std::max(max_depth, cluster_max_depth[i]);
            mse_sum += cluster_mse_sum[i];
            if (cluster_real[i] > 0) {
                mse_min = std::min(mse_min, cluster_mse_min[i]);
                mse_max = std::max(mse_max, cluster_mse_max[i]);
            }
        }

        if (profile_timing) {
            const double t1 = wall_now_s();
            t_wall_reduce = t1 - t_wall_t0;
            t_wall_t0 = t1;
        }

        double t_read_ivf = 0.0;
        double t_read_raw = 0.0;
        double t_read_base = 0.0;
        double t_wait_async_pop = 0.0;
        double t_bad = 0.0;
        double t_linkage = 0.0;
        double t_write_sum = 0.0;
        double t_read_ivf_max = 0.0;
        double t_read_raw_max = 0.0;
        double t_read_base_max = 0.0;
        double t_wait_async_pop_max = 0.0;
        double t_bad_max = 0.0;
        double t_linkage_max = 0.0;
        double t_write_max = 0.0;

        double t_bad_knn_max = 0.0;
        double t_bad_umap_max = 0.0;
        double t_bad_umap_rho_sigma_max = 0.0;
        double t_bad_umap_build_pdir_max = 0.0;
        double t_bad_umap_wsym_deg_max = 0.0;
        double t_bad_umap_seed_select_max = 0.0;
        double t_bad_umap_barycenter_max = 0.0;
        double t_bad_umap_encode_max = 0.0;
        double t_bad_umap_encode_gemm_max = 0.0;
        double t_bad_umap_encode_greedy_max = 0.0;
        double t_bad_umap_encode_icm_max = 0.0;
        double t_bad_umap_encode_recon_max = 0.0;
        double t_bad_umap_encode_max_cluster = 0.0;
        int t_bad_umap_encode_max_cluster_cid = -1;
        int t_bad_umap_encode_max_cluster_n_real = 0;
        int t_bad_umap_encode_max_cluster_k_virtual = 0;
        int t_bad_umap_encode_max_cluster_n_centers = 0;
        double t_bad_dyn_window_tiny_cpu_max = 0.0;
        double t_cand_build_cpu_max = 0.0;
        double t_gpu_wait_max = 0.0;
        double t_rfull_update_max = 0.0;
        double t_cpu_fallback_max = 0.0;
        double t_commit_cpu_max = 0.0;
        double t_cpu_quantize_max = 0.0;
        double t_cpu_ls_max = 0.0;
        double t_cpu_icm_max = 0.0;
        double t_cpu_cost_max = 0.0;
        double t_linkage_good_max = 0.0;
        double t_linkage_bad_max = 0.0;

        std::uint64_t cnt_gpu_many = 0;
        std::uint64_t cnt_gpu_batch = 0;
        std::uint64_t cnt_gpu_single = 0;
        std::uint64_t cnt_rfull_updates = 0;
        std::uint64_t cnt_rfull_update_batches = 0;

        double t_good_cand_build_cpu_max = 0.0;
        double t_good_hnsw_build_cpu_max = 0.0;
        double t_good_hnsw_query_cpu_max = 0.0;
        double t_good_cand_pack_cpu_max = 0.0;
        double t_good_gpu_wait_max = 0.0;
        double t_good_rfull_update_max = 0.0;
        double t_good_cpu_fallback_max = 0.0;
        double t_good_commit_cpu_max = 0.0;
        double t_good_dyn_window_tiny_cpu_max = 0.0;
        std::uint64_t cnt_good_gpu_many = 0;
        std::uint64_t cnt_good_gpu_batch = 0;
        std::uint64_t cnt_good_gpu_single = 0;
        std::uint64_t cnt_good_rfull_updates = 0;
        std::uint64_t cnt_good_rfull_update_batches = 0;
        std::uint64_t cnt_good_dyn_window_tiny_cpu_calls = 0;
        std::uint64_t cnt_good_dyn_window_tiny_cpu_nodes_sum = 0;
        std::uint64_t cnt_good_dyn_window_tiny_cpu_pairs_sum = 0;
        std::uint64_t cnt_good_dyn_window_preflush_calls = 0;
        std::uint64_t cnt_good_dyn_window_preflush_nodes_sum = 0;
        std::uint64_t cnt_good_dyn_window_preflush_pairs_sum = 0;
        std::uint64_t cnt_good_dyn_window_forced_calls = 0;
        std::uint64_t cnt_good_dyn_window_forced_nodes_sum = 0;
        std::uint64_t cnt_good_dyn_window_forced_pairs_sum = 0;
        std::uint64_t cnt_good_fb_nodes = 0;
        std::uint64_t cnt_good_fb_tiny_dyn_window = 0;
        std::uint64_t cnt_good_fb_no_cuda = 0;
        std::uint64_t cnt_good_fb_kp_small = 0;
        std::uint64_t cnt_good_fb_kp_large = 0;
        std::uint64_t cnt_good_fb_dev_rfull_disabled = 0;
        std::uint64_t cnt_good_fb_gpu_error = 0;
        std::uint64_t cnt_good_fb_same_layer = 0;
        std::uint64_t cnt_good_fb_other = 0;
        std::uint64_t cnt_bad_dyn_window_tiny_cpu_calls = 0;
        std::uint64_t cnt_bad_dyn_window_tiny_cpu_nodes_sum = 0;
        std::uint64_t cnt_bad_dyn_window_tiny_cpu_pairs_sum = 0;
        std::array<std::uint64_t, 12> cnt_good_dyn_kp_hist{};
        int cnt_good_dyn_kp_max = 0;
        std::array<std::uint64_t, 4> cnt_good_mn_calls_by_src{};
        std::array<std::uint64_t, 4> cnt_good_mn_nodes_sum_by_src{};
        std::array<std::uint64_t, 4> cnt_good_mn_pairs_sum_by_src{};
        std::array<double, 4> t_good_gpu_wait_mn_by_src_max{};
        std::array<std::uint64_t, 4> cnt_good_gpu_wait_mn_calls_by_src{};
        double t_good_gpu_wait_batch_max = 0.0;
        double t_good_gpu_wait_single_max = 0.0;
        std::array<std::uint64_t, 12> cnt_good_mn_npairs_hist{};
        std::array<std::uint64_t, 12> cnt_good_mn_nodes_hist{};
        int cnt_good_mn_npairs_max = 0;
        int cnt_good_mn_nodes_max = 0;

        for (const auto& tt : timing_tls) {
            t_read_ivf += tt.read_ivf;
            t_read_raw += tt.read_raw;
            t_read_base += tt.read_base;
            t_wait_async_pop += tt.wait_async_pop;
            t_bad += tt.bad_knn_umap;
            t_linkage += tt.linkage_core;
            t_write_sum += tt.write;
            t_read_ivf_max = std::max(t_read_ivf_max, tt.read_ivf);
            t_read_raw_max = std::max(t_read_raw_max, tt.read_raw);
            t_read_base_max = std::max(t_read_base_max, tt.read_base);
            t_wait_async_pop_max = std::max(t_wait_async_pop_max, tt.wait_async_pop);
            t_bad_max = std::max(t_bad_max, tt.bad_knn_umap);
            t_linkage_max = std::max(t_linkage_max, tt.linkage_core);
            t_write_max = std::max(t_write_max, tt.write);

            t_bad_knn_max = std::max(t_bad_knn_max, tt.prof.bad_knn_s);
            t_bad_umap_max = std::max(t_bad_umap_max, tt.prof.bad_umap_s);
            t_bad_umap_rho_sigma_max = std::max(t_bad_umap_rho_sigma_max, tt.prof.bad_umap_rho_sigma_s);
            t_bad_umap_build_pdir_max = std::max(t_bad_umap_build_pdir_max, tt.prof.bad_umap_build_pdir_s);
            t_bad_umap_wsym_deg_max = std::max(t_bad_umap_wsym_deg_max, tt.prof.bad_umap_wsym_deg_s);
            t_bad_umap_seed_select_max = std::max(t_bad_umap_seed_select_max, tt.prof.bad_umap_seed_select_s);
            t_bad_umap_barycenter_max = std::max(t_bad_umap_barycenter_max, tt.prof.bad_umap_barycenter_s);
            t_bad_umap_encode_max = std::max(t_bad_umap_encode_max, tt.prof.bad_umap_encode_s);
            t_bad_umap_encode_gemm_max = std::max(t_bad_umap_encode_gemm_max, tt.prof.bad_umap_encode_gemm_s);
            t_bad_umap_encode_greedy_max = std::max(t_bad_umap_encode_greedy_max, tt.prof.bad_umap_encode_greedy_s);
            t_bad_umap_encode_icm_max = std::max(t_bad_umap_encode_icm_max, tt.prof.bad_umap_encode_icm_s);
            t_bad_umap_encode_recon_max = std::max(t_bad_umap_encode_recon_max, tt.prof.bad_umap_encode_recon_s);
            t_bad_dyn_window_tiny_cpu_max =
                std::max(t_bad_dyn_window_tiny_cpu_max, tt.prof.bad_dyn_window_tiny_cpu_s);
            if (tt.prof.bad_umap_encode_max_cluster_s > t_bad_umap_encode_max_cluster) {
                t_bad_umap_encode_max_cluster = tt.prof.bad_umap_encode_max_cluster_s;
                t_bad_umap_encode_max_cluster_cid = tt.prof.bad_umap_encode_max_cluster_cid;
                t_bad_umap_encode_max_cluster_n_real = tt.prof.bad_umap_encode_max_cluster_n_real;
                t_bad_umap_encode_max_cluster_k_virtual = tt.prof.bad_umap_encode_max_cluster_k_virtual;
                t_bad_umap_encode_max_cluster_n_centers = tt.prof.bad_umap_encode_max_cluster_n_centers;
            }
            t_cand_build_cpu_max = std::max(t_cand_build_cpu_max, tt.prof.cand_build_cpu_s);
            t_gpu_wait_max = std::max(t_gpu_wait_max, tt.prof.gpu_wait_s);
            t_rfull_update_max = std::max(t_rfull_update_max, tt.prof.rfull_update_s);
            t_cpu_fallback_max = std::max(t_cpu_fallback_max, tt.prof.cpu_fallback_s);
            t_commit_cpu_max = std::max(t_commit_cpu_max, tt.prof.commit_cpu_s);
            t_cpu_quantize_max = std::max(t_cpu_quantize_max, tt.prof.cpu_quantize_s);
            t_cpu_ls_max = std::max(t_cpu_ls_max, tt.prof.cpu_ls_s);
            t_cpu_icm_max = std::max(t_cpu_icm_max, tt.prof.cpu_icm_s);
            t_cpu_cost_max = std::max(t_cpu_cost_max, tt.prof.cpu_cost_s);
            t_linkage_good_max = std::max(t_linkage_good_max, tt.prof.linkage_good_s);
            t_linkage_bad_max = std::max(t_linkage_bad_max, tt.prof.linkage_bad_s);

            cnt_gpu_many += tt.prof.gpu_calls_many_nodes;
            cnt_gpu_batch += tt.prof.gpu_calls_batch;
            cnt_gpu_single += tt.prof.gpu_calls_single;
            cnt_rfull_updates += tt.prof.rfull_update_calls;
            cnt_rfull_update_batches += tt.prof.rfull_update_batches;

            t_good_cand_build_cpu_max = std::max(t_good_cand_build_cpu_max, tt.prof.good_cand_build_cpu_s);
            t_good_hnsw_build_cpu_max = std::max(t_good_hnsw_build_cpu_max, tt.prof.good_hnsw_build_cpu_s);
            t_good_hnsw_query_cpu_max = std::max(t_good_hnsw_query_cpu_max, tt.prof.good_hnsw_query_cpu_s);
            t_good_cand_pack_cpu_max = std::max(t_good_cand_pack_cpu_max, tt.prof.good_cand_pack_cpu_s);
            t_good_gpu_wait_max = std::max(t_good_gpu_wait_max, tt.prof.good_gpu_wait_s);
            for (int i = 0; i < 4; ++i) {
                t_good_gpu_wait_mn_by_src_max[static_cast<std::size_t>(i)] =
                    std::max(t_good_gpu_wait_mn_by_src_max[static_cast<std::size_t>(i)],
                             tt.prof.good_gpu_wait_mn_s_by_src[i]);
                cnt_good_gpu_wait_mn_calls_by_src[static_cast<std::size_t>(i)] +=
                    tt.prof.good_gpu_wait_mn_calls_by_src[i];
            }
            t_good_gpu_wait_batch_max = std::max(t_good_gpu_wait_batch_max, tt.prof.good_gpu_wait_batch_s);
            t_good_gpu_wait_single_max = std::max(t_good_gpu_wait_single_max, tt.prof.good_gpu_wait_single_s);
            t_good_rfull_update_max = std::max(t_good_rfull_update_max, tt.prof.good_rfull_update_s);
            t_good_cpu_fallback_max = std::max(t_good_cpu_fallback_max, tt.prof.good_cpu_fallback_s);
            t_good_commit_cpu_max = std::max(t_good_commit_cpu_max, tt.prof.good_commit_cpu_s);
            t_good_dyn_window_tiny_cpu_max =
                std::max(t_good_dyn_window_tiny_cpu_max, tt.prof.good_dyn_window_tiny_cpu_s);
            cnt_good_gpu_many += tt.prof.good_gpu_calls_many_nodes;
            cnt_good_gpu_batch += tt.prof.good_gpu_calls_batch;
            cnt_good_gpu_single += tt.prof.good_gpu_calls_single;
            cnt_good_rfull_updates += tt.prof.good_rfull_update_calls;
            cnt_good_rfull_update_batches += tt.prof.good_rfull_update_batches;
            cnt_good_dyn_window_tiny_cpu_calls += tt.prof.good_dyn_window_tiny_cpu_calls;
            cnt_good_dyn_window_tiny_cpu_nodes_sum += tt.prof.good_dyn_window_tiny_cpu_nodes_sum;
            cnt_good_dyn_window_tiny_cpu_pairs_sum += tt.prof.good_dyn_window_tiny_cpu_pairs_sum;
            cnt_good_dyn_window_preflush_calls += tt.prof.good_dyn_window_preflush_calls;
            cnt_good_dyn_window_preflush_nodes_sum += tt.prof.good_dyn_window_preflush_nodes_sum;
            cnt_good_dyn_window_preflush_pairs_sum += tt.prof.good_dyn_window_preflush_pairs_sum;
            cnt_good_dyn_window_forced_calls += tt.prof.good_dyn_window_forced_calls;
            cnt_good_dyn_window_forced_nodes_sum += tt.prof.good_dyn_window_forced_nodes_sum;
            cnt_good_dyn_window_forced_pairs_sum += tt.prof.good_dyn_window_forced_pairs_sum;
            cnt_good_fb_nodes += tt.prof.good_cpu_fallback_nodes;
            cnt_good_fb_tiny_dyn_window += tt.prof.good_cpu_fallback_dyn_window_tiny_cpu;
            cnt_good_fb_no_cuda += tt.prof.good_cpu_fallback_no_cuda;
            cnt_good_fb_kp_small += tt.prof.good_cpu_fallback_kp_too_small;
            cnt_good_fb_kp_large += tt.prof.good_cpu_fallback_kp_too_large;
            cnt_good_fb_dev_rfull_disabled += tt.prof.good_cpu_fallback_device_rfull_disabled;
            cnt_good_fb_gpu_error += tt.prof.good_cpu_fallback_gpu_error;
            cnt_good_fb_same_layer += tt.prof.good_cpu_fallback_same_layer_present;
            cnt_good_fb_other += tt.prof.good_cpu_fallback_other;
            cnt_bad_dyn_window_tiny_cpu_calls += tt.prof.bad_dyn_window_tiny_cpu_calls;
            cnt_bad_dyn_window_tiny_cpu_nodes_sum += tt.prof.bad_dyn_window_tiny_cpu_nodes_sum;
            cnt_bad_dyn_window_tiny_cpu_pairs_sum += tt.prof.bad_dyn_window_tiny_cpu_pairs_sum;
            for (int i = 0; i < 12; ++i) {
                cnt_good_dyn_kp_hist[static_cast<std::size_t>(i)] += tt.prof.good_dyn_kp_hist[i];
            }
            cnt_good_dyn_kp_max = std::max(cnt_good_dyn_kp_max, tt.prof.good_dyn_kp_max);
            for (int i = 0; i < 4; ++i) {
                cnt_good_mn_calls_by_src[static_cast<std::size_t>(i)] += tt.prof.good_mn_calls_by_src[i];
                cnt_good_mn_nodes_sum_by_src[static_cast<std::size_t>(i)] += tt.prof.good_mn_nodes_sum_by_src[i];
                cnt_good_mn_pairs_sum_by_src[static_cast<std::size_t>(i)] += tt.prof.good_mn_pairs_sum_by_src[i];
            }
            for (int i = 0; i < 12; ++i) {
                cnt_good_mn_npairs_hist[static_cast<std::size_t>(i)] += tt.prof.good_mn_npairs_hist[i];
                cnt_good_mn_nodes_hist[static_cast<std::size_t>(i)] += tt.prof.good_mn_nodes_hist[i];
            }
            cnt_good_mn_npairs_max = std::max(cnt_good_mn_npairs_max, tt.prof.good_mn_npairs_max);
            cnt_good_mn_nodes_max = std::max(cnt_good_mn_nodes_max, tt.prof.good_mn_nodes_max);
        }
        if (cfg.large.profile_timing) {
            LogInfo("Linkage streaming wall breakdown(s): plan_open=" + FormatFloatLocal(t_wall_pass_a, 6) +
                " coeff_codec_pre=" + FormatFloatLocal(t_wall_coeff_pre, 6) +
                " cluster_build=" + FormatFloatLocal(t_wall_pass_b, 6) +
                " linkage_list_finish=" + FormatFloatLocal(t_wall_writer_finish, 6) +
                " coeff_codec_finish=" + FormatFloatLocal(t_wall_coeff_finish, 6) +
                " summarize=" + FormatFloatLocal(t_wall_reduce, 6));
            LogInfo("Linkage streaming timing (sum over threads): read_ivf=" + FormatFloatLocal(t_read_ivf, 6) +
                " read_raw=" + FormatFloatLocal(t_read_raw, 6) +
                " read_base=" + FormatFloatLocal(t_read_base, 6) +
                " wait_async_pop=" + FormatFloatLocal(t_wait_async_pop, 6) +
                " bad_knn_umap=" + FormatFloatLocal(t_bad, 6) +
                " linkage_core_total=" + FormatFloatLocal(t_linkage, 6) +
                " write=" + FormatFloatLocal(t_write_sum, 6));
            LogInfo("Linkage streaming timing (wall, max thread): read_ivf=" + FormatFloatLocal(t_read_ivf_max, 6) +
                " read_raw=" + FormatFloatLocal(t_read_raw_max, 6) +
                " read_base=" + FormatFloatLocal(t_read_base_max, 6) +
                " wait_async_pop=" + FormatFloatLocal(t_wait_async_pop_max, 6) +
                " bad_knn_umap=" + FormatFloatLocal(t_bad_max, 6) +
                " linkage_core_total=" + FormatFloatLocal(t_linkage_max, 6) +
                " write=" + FormatFloatLocal(t_write_max, 6));
            LogInfo("Linkage core bad breakdown (wall, max thread): knn=" + FormatFloatLocal(t_bad_knn_max, 6) +
                " umap=" + FormatFloatLocal(t_bad_umap_max, 6) +
                " cand_build_cpu=" + FormatFloatLocal(t_cand_build_cpu_max, 6) +
                " gpu_wait=" + FormatFloatLocal(t_gpu_wait_max, 6) +
                " rfull_update=" + FormatFloatLocal(t_rfull_update_max, 6) +
                " cpu_fallback=" + FormatFloatLocal(t_cpu_fallback_max, 6) +
                " tiny_dyn_window_in_cpu_fallback=" + FormatFloatLocal(t_bad_dyn_window_tiny_cpu_max, 6) +
                " commit_cpu=" + FormatFloatLocal(t_commit_cpu_max, 6));
            LogInfo("Linkage core bad_umap breakdown (wall, max thread): rho_sigma=" + FormatFloatLocal(
                    t_bad_umap_rho_sigma_max, 6) +
                " pdir=" + FormatFloatLocal(t_bad_umap_build_pdir_max, 6) +
                " wsym_deg=" + FormatFloatLocal(t_bad_umap_wsym_deg_max, 6) +
                " seeds=" + FormatFloatLocal(t_bad_umap_seed_select_max, 6) +
                " bary=" + FormatFloatLocal(t_bad_umap_barycenter_max, 6) +
                " encode=" + FormatFloatLocal(t_bad_umap_encode_max, 6));
            LogInfo("Linkage core bad_umap encode breakdown (wall, max thread): gemm=" + FormatFloatLocal(
                    t_bad_umap_encode_gemm_max, 6) +
                " greedy=" + FormatFloatLocal(t_bad_umap_encode_greedy_max, 6) +
                " icm=" + FormatFloatLocal(t_bad_umap_encode_icm_max, 6) +
                " recon=" + FormatFloatLocal(t_bad_umap_encode_recon_max, 6));
            if (t_bad_umap_encode_max_cluster > 0.0) {
                LogInfo("Linkage core bad_umap encode worst cluster: cid=" + std::to_string(
                        t_bad_umap_encode_max_cluster_cid) +
                    " n_real=" + std::to_string(t_bad_umap_encode_max_cluster_n_real) +
                    " k_virtual=" + std::to_string(t_bad_umap_encode_max_cluster_k_virtual) +
                    " n_centers=" + std::to_string(t_bad_umap_encode_max_cluster_n_centers) +
                    " t=" + FormatFloatLocal(t_bad_umap_encode_max_cluster, 6));
            }
            LogInfo("Linkage core wall by cluster type (max thread): good=" + FormatFloatLocal(t_linkage_good_max, 6) +
                " bad=" + FormatFloatLocal(t_linkage_bad_max, 6));
            LogInfo("Linkage core good breakdown (wall, max thread): good_hnsw_build_cpu=" + FormatFloatLocal(
                    t_good_hnsw_build_cpu_max, 6) +
                " good_hnsw_query_cpu=" + FormatFloatLocal(t_good_hnsw_query_cpu_max, 6) +
                " good_cand_pack_cpu=" + FormatFloatLocal(t_good_cand_pack_cpu_max, 6) +
                " good_cand_build_cpu=" + FormatFloatLocal(t_good_cand_build_cpu_max, 6) +
                " gpu_wait=" + FormatFloatLocal(t_good_gpu_wait_max, 6) +
                " rfull_update=" + FormatFloatLocal(t_good_rfull_update_max, 6) +
                " cpu_fallback=" + FormatFloatLocal(t_good_cpu_fallback_max, 6) +
                " tiny_dyn_window_in_cpu_fallback=" + FormatFloatLocal(t_good_dyn_window_tiny_cpu_max, 6) +
                " commit_cpu=" + FormatFloatLocal(t_good_commit_cpu_max, 6));
            LogInfo("Linkage core good breakdown (counts): gpu_many_nodes=" + std::to_string(cnt_good_gpu_many) +
                " gpu_batch=" + std::to_string(cnt_good_gpu_batch) +
                " gpu_single=" + std::to_string(cnt_good_gpu_single) +
                " rfull_updates=" + std::to_string(cnt_good_rfull_updates) +
                " rfull_update_batches=" + std::to_string(cnt_good_rfull_update_batches) +
                " avg_cols_per_batch=" +
                FormatFloatLocal((cnt_good_rfull_update_batches > 0)
                                     ? (static_cast<double>(cnt_good_rfull_updates) /
                                         static_cast<double>(cnt_good_rfull_update_batches))
                                     : 0.0,
                                 3));
            {
                std::uint64_t mn_calls = 0;
                for (std::uint64_t c : cnt_good_gpu_wait_mn_calls_by_src) mn_calls += c;
                if (mn_calls > 0 || t_good_gpu_wait_batch_max > 0.0 || t_good_gpu_wait_single_max > 0.0) {
                    LogInfo("Linkage core good gpu_wait split (wall, max thread):"
                        " mn_inner=" + FormatFloatLocal(t_good_gpu_wait_mn_by_src_max[0], 6) +
                        " mn_same_frozen=" + FormatFloatLocal(t_good_gpu_wait_mn_by_src_max[1], 6) +
                        " mn_dyn_before=" + FormatFloatLocal(t_good_gpu_wait_mn_by_src_max[2], 6) +
                        " mn_dyn_window=" + FormatFloatLocal(t_good_gpu_wait_mn_by_src_max[3], 6) +
                        " batch=" + FormatFloatLocal(t_good_gpu_wait_batch_max, 6) +
                        " single=" + FormatFloatLocal(t_good_gpu_wait_single_max, 6) +
                        " (mn_calls: inner=" + std::to_string(cnt_good_gpu_wait_mn_calls_by_src[0]) +
                        " same_frozen=" + std::to_string(cnt_good_gpu_wait_mn_calls_by_src[1]) +
                        " dyn_before=" + std::to_string(cnt_good_gpu_wait_mn_calls_by_src[2]) +
                        " dyn_window=" + std::to_string(cnt_good_gpu_wait_mn_calls_by_src[3]) + ")");
                }
            }
            if (cnt_good_dyn_window_tiny_cpu_calls > 0) {
                const double avg_pairs =
                    static_cast<double>(cnt_good_dyn_window_tiny_cpu_pairs_sum) /
                    static_cast<double>(cnt_good_dyn_window_tiny_cpu_calls);
                const double avg_nodes =
                    static_cast<double>(cnt_good_dyn_window_tiny_cpu_nodes_sum) /
                    static_cast<double>(cnt_good_dyn_window_tiny_cpu_calls);
                LogInfo("Linkage core good dyn_window tiny_cpu (counts): calls=" +
                    std::to_string(cnt_good_dyn_window_tiny_cpu_calls) +
                    " avg_pairs=" + FormatFloatLocal(avg_pairs, 3) +
                    " avg_nodes=" + FormatFloatLocal(avg_nodes, 3));
            }
            if (cnt_good_dyn_window_preflush_calls > 0 || cnt_good_dyn_window_forced_calls > 0) {
                const auto avg = [](std::uint64_t sum, std::uint64_t calls) -> double
                {
                    return (calls > 0) ? (static_cast<double>(sum) / static_cast<double>(calls)) : 0.0;
                };
                LogInfo("Linkage core good dyn_window GPU flush split (counts): preflush_calls=" +
                    std::to_string(cnt_good_dyn_window_preflush_calls) +
                    " preflush_avg_pairs=" +
                    FormatFloatLocal(avg(cnt_good_dyn_window_preflush_pairs_sum, cnt_good_dyn_window_preflush_calls),
                                     3) +
                    " preflush_avg_nodes=" +
                    FormatFloatLocal(avg(cnt_good_dyn_window_preflush_nodes_sum, cnt_good_dyn_window_preflush_calls),
                                     3) +
                    " forced_calls=" + std::to_string(cnt_good_dyn_window_forced_calls) +
                    " forced_avg_pairs=" +
                    FormatFloatLocal(avg(cnt_good_dyn_window_forced_pairs_sum, cnt_good_dyn_window_forced_calls), 3) +
                    " forced_avg_nodes=" +
                    FormatFloatLocal(avg(cnt_good_dyn_window_forced_nodes_sum, cnt_good_dyn_window_forced_calls), 3));
            }
        }
        if (cnt_bad_dyn_window_tiny_cpu_calls > 0) {
            const double avg_pairs =
                static_cast<double>(cnt_bad_dyn_window_tiny_cpu_pairs_sum) /
                static_cast<double>(cnt_bad_dyn_window_tiny_cpu_calls);
            const double avg_nodes =
                static_cast<double>(cnt_bad_dyn_window_tiny_cpu_nodes_sum) /
                static_cast<double>(cnt_bad_dyn_window_tiny_cpu_calls);
            LogInfo("Linkage core bad dyn_window tiny_cpu (counts): calls=" +
                std::to_string(cnt_bad_dyn_window_tiny_cpu_calls) +
                " avg_pairs=" + FormatFloatLocal(avg_pairs, 3) +
                " avg_nodes=" + FormatFloatLocal(avg_nodes, 3));
        }
        if (cnt_good_fb_nodes > 0) {
            LogInfo("Linkage core good CPU fallback reasons (counts): total=" + std::to_string(cnt_good_fb_nodes) +
                " tiny_dyn_window=" + std::to_string(cnt_good_fb_tiny_dyn_window) +
                " no_cuda=" + std::to_string(cnt_good_fb_no_cuda) +
                " kp_small=" + std::to_string(cnt_good_fb_kp_small) +
                " kp_large=" + std::to_string(cnt_good_fb_kp_large) +
                " device_rfull_disabled=" + std::to_string(cnt_good_fb_dev_rfull_disabled) +
                " gpu_error=" + std::to_string(cnt_good_fb_gpu_error) +
                " same_layer_present=" + std::to_string(cnt_good_fb_same_layer) +
                " other=" + std::to_string(cnt_good_fb_other));
        }
        {
            std::uint64_t total = 0;
            for (std::uint64_t c : cnt_good_dyn_kp_hist) total += c;
            if (total > 0) {
                static const int k_bucket_hi[12] = {0, 1, 2, 3, 4, 7, 15, 31, 63, 127, 255, 256};
                const auto approx_pct = [&](double p) -> int
                {
                    const auto target =
                        static_cast<std::uint64_t>(std::ceil(p * static_cast<double>(total)));
                    std::uint64_t cum = 0;
                    for (int b = 0; b < 12; ++b) {
                        cum += cnt_good_dyn_kp_hist[static_cast<std::size_t>(b)];
                        if (cum >= target) {
                            if (b == 11) return std::max(256, cnt_good_dyn_kp_max);
                            return k_bucket_hi[b];
                        }
                    }
                    return cnt_good_dyn_kp_max;
                };
                const int p50 = approx_pct(0.50);
                const int p90 = approx_pct(0.90);
                LogInfo("Linkage core good same-layer dynamic Kp (nodes): n=" + std::to_string(total) +
                    " p50~" + std::to_string(p50) +
                    " p90~" + std::to_string(p90) +
                    " max=" + std::to_string(cnt_good_dyn_kp_max) +
                    " (buckets: 0,1,2,3,4,5-7,8-15,16-31,32-63,64-127,128-255,256+)");
                LogInfo("Linkage core good same-layer dynamic Kp hist (nodes): "
                    "0=" + std::to_string(cnt_good_dyn_kp_hist[0]) +
                    " 1=" + std::to_string(cnt_good_dyn_kp_hist[1]) +
                    " 2=" + std::to_string(cnt_good_dyn_kp_hist[2]) +
                    " 3=" + std::to_string(cnt_good_dyn_kp_hist[3]) +
                    " 4=" + std::to_string(cnt_good_dyn_kp_hist[4]) +
                    " 5-7=" + std::to_string(cnt_good_dyn_kp_hist[5]) +
                    " 8-15=" + std::to_string(cnt_good_dyn_kp_hist[6]) +
                    " 16-31=" + std::to_string(cnt_good_dyn_kp_hist[7]) +
                    " 32-63=" + std::to_string(cnt_good_dyn_kp_hist[8]) +
                    " 64-127=" + std::to_string(cnt_good_dyn_kp_hist[9]) +
                    " 128-255=" + std::to_string(cnt_good_dyn_kp_hist[10]) +
                    " 256+=" + std::to_string(cnt_good_dyn_kp_hist[11]));
            }
        }
        {
            std::uint64_t total_calls = 0;
            std::uint64_t total_pairs = 0;
            std::uint64_t total_nodes = 0;
            for (int i = 0; i < 4; ++i) {
                total_calls += cnt_good_mn_calls_by_src[static_cast<std::size_t>(i)];
                total_pairs += cnt_good_mn_pairs_sum_by_src[static_cast<std::size_t>(i)];
                total_nodes += cnt_good_mn_nodes_sum_by_src[static_cast<std::size_t>(i)];
            }
            if (total_calls > 0) {
                const auto avg = [](std::uint64_t sum, std::uint64_t den) -> double
                {
                    if (den == 0) return 0.0;
                    return static_cast<double>(sum) / static_cast<double>(den);
                };
                static const int k_pairs_hi[12] = {
                    0, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535, 65536
                };
                static const int k_nodes_hi[12] = {
                    0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 513
                };
                const auto approx_pct_from_hist =
                    [&](const std::array<std::uint64_t, 12>& hist,
                        int hist_max,
                        const int (&bucket_hi)[12],
                        double p) -> int
                {
                    std::uint64_t tot = 0;
                    for (std::uint64_t c : hist) tot += c;
                    if (tot == 0) return 0;
                    const auto target =
                        static_cast<std::uint64_t>(std::ceil(p * static_cast<double>(tot)));
                    std::uint64_t cum = 0;
                    for (int b = 0; b < 12; ++b) {
                        cum += hist[static_cast<std::size_t>(b)];
                        if (cum >= target) {
                            if (b == 11) return std::max(bucket_hi[11], hist_max);
                            return bucket_hi[b];
                        }
                    }
                    return hist_max;
                };
                const int p50_pairs =
                    approx_pct_from_hist(cnt_good_mn_npairs_hist, cnt_good_mn_npairs_max, k_pairs_hi, 0.50);
                const int p90_pairs =
                    approx_pct_from_hist(cnt_good_mn_npairs_hist, cnt_good_mn_npairs_max, k_pairs_hi, 0.90);
                const int p50_nodes =
                    approx_pct_from_hist(cnt_good_mn_nodes_hist, cnt_good_mn_nodes_max, k_nodes_hi, 0.50);
                const int p90_nodes =
                    approx_pct_from_hist(cnt_good_mn_nodes_hist, cnt_good_mn_nodes_max, k_nodes_hi, 0.90);

                LogInfo("Linkage core good many-nodes batch quality: calls=" + std::to_string(total_calls) +
                    " avg_pairs=" + FormatFloatLocal(avg(total_pairs, total_calls), 3) +
                    " p50~" + std::to_string(p50_pairs) +
                    " p90~" + std::to_string(p90_pairs) +
                    " avg_nodes=" + FormatFloatLocal(avg(total_nodes, total_calls), 3) +
                    " p50~" + std::to_string(p50_nodes) +
                    " p90~" + std::to_string(p90_nodes));
                const auto src_line = [&](const char* name, int idx) -> std::string
                {
                    const std::uint64_t c = cnt_good_mn_calls_by_src[static_cast<std::size_t>(idx)];
                    const std::uint64_t sp = cnt_good_mn_pairs_sum_by_src[static_cast<std::size_t>(idx)];
                    const std::uint64_t sn = cnt_good_mn_nodes_sum_by_src[static_cast<std::size_t>(idx)];
                    return std::string(name) + ": calls=" + std::to_string(c) +
                        " avg_pairs=" + FormatFloatLocal(avg(sp, c), 3) +
                        " avg_nodes=" + FormatFloatLocal(avg(sn, c), 3);
                };
                LogInfo("Linkage core good many-nodes sources: " +
                    src_line("inner", 0) + " " +
                    src_line("same_frozen", 1) + " " +
                    src_line("dyn_before", 2) + " " +
                    src_line("dyn_window", 3));
            }
        }
        if (cfg.large.profile_timing) {
            const bool meaningful =
                (t_cpu_quantize_max > 0.0) || (t_cpu_ls_max > 0.0) || (t_cpu_icm_max > 0.0) ||
                (t_cpu_cost_max > 0.0) || (cnt_gpu_many > 0) || (cnt_gpu_batch > 0) || (cnt_gpu_single > 0) ||
                (cnt_rfull_updates > 0) || (cnt_rfull_update_batches > 0);
            if (meaningful) {
                static std::atomic<bool> did_log{false};
                bool expected = false;
                if (did_log.compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
                    LogInfo("Linkage core bad cpu_eval breakdown (wall, max thread): quantize=" +
                        FormatFloatLocal(t_cpu_quantize_max, 6) +
                        " ls=" + FormatFloatLocal(t_cpu_ls_max, 6) +
                        " icm=" + FormatFloatLocal(t_cpu_icm_max, 6) +
                        " cost=" + FormatFloatLocal(t_cpu_cost_max, 6));
                    LogInfo("Linkage core bad breakdown (counts): gpu_many_nodes=" + std::to_string(cnt_gpu_many) +
                        " gpu_batch=" + std::to_string(cnt_gpu_batch) +
                        " gpu_single=" + std::to_string(cnt_gpu_single) +
                        " rfull_updates=" + std::to_string(cnt_rfull_updates) +
                        " rfull_update_batches=" + std::to_string(cnt_rfull_update_batches) +
                        " avg_cols_per_batch=" +
                        FormatFloatLocal((cnt_rfull_update_batches > 0)
                                             ? (static_cast<double>(cnt_rfull_updates) /
                                                 static_cast<double>(cnt_rfull_update_batches))
                                             : 0.0,
                                         3));
                }
            }
        }

#if defined(STLQ_ENABLE_CUDA)
        if (cfg.runtime.use_cuda) {
            const CudaLinkageEvalStats stats = GetAndResetCudaLinkageEvalStats();
            if (stats.calls > 0) {
                LogInfo(
                    "CUDA linkage candidate-eval timing (sum over threads, total): calls=" + std::to_string(stats.calls)
                    +
                    " h2d=" + FormatFloatLocal(stats.h2d_s, 6) +
                    " residual=" + FormatFloatLocal(stats.residual_s, 6) +
                    " encode=" + FormatFloatLocal(stats.encode_s, 6) +
                    " reduce=" + FormatFloatLocal(stats.reduce_s, 6) +
                    " d2h=" + FormatFloatLocal(stats.d2h_s, 6) +
                    " total=" + FormatFloatLocal(stats.total_s, 6));

                if (cfg.large.profile_timing) {
                    const CudaLinkageEncodeStats enc = GetAndResetCudaLinkageEncodeStats();
                    const CudaLinkageEncodeLargeRootStats lr = GetAndResetCudaLinkageEncodeLargeRootStats();
                    LogInfo(
                        "CUDA linkage encode breakdown (sum over calls, total): calls=" + std::to_string(enc.calls) +
                        " gemm=" + FormatFloatLocal(enc.gemm_s, 6) +
                        " greedy=" + FormatFloatLocal(enc.greedy_s, 6) +
                        " ls_cost=" + FormatFloatLocal(enc.ls_cost_s, 6) +
                        " icm=" + FormatFloatLocal(enc.icm_s, 6) +
                        " ils=" + FormatFloatLocal(enc.ils_s, 6) +
                        " (copy=" + FormatFloatLocal(enc.ils_copy_s, 6) +
                        " solve_cost=" + FormatFloatLocal(enc.ils_solve_cost_s, 6) +
                        " icm=" + FormatFloatLocal(enc.ils_icm_s, 6) +
                        " enc=" + FormatFloatLocal(enc.ils_encode_s, 6) +
                        " acc=" + FormatFloatLocal(enc.ils_accept_s, 6) + ")" +
                        " total=" + FormatFloatLocal(enc.total_s, 6));
                    if (enc.calls > 0) {
                        const double inv = 1000.0 / static_cast<double>(enc.calls);
                        LogInfo("CUDA linkage encode avg (ms/call, total): total=" + FormatFloatLocal(
                                enc.total_s * inv, 6) +
                            " gemm=" + FormatFloatLocal(enc.gemm_s * inv, 6) +
                            " greedy=" + FormatFloatLocal(enc.greedy_s * inv, 6) +
                            " ls_cost=" + FormatFloatLocal(enc.ls_cost_s * inv, 6) +
                            " icm=" + FormatFloatLocal(enc.icm_s * inv, 6) +
                            " ils_icm=" + FormatFloatLocal(enc.ils_icm_s * inv, 6));
                    }
                    if (lr.calls > 0) {
                        const double avg_root_chunks = static_cast<double>(lr.root_chunks_total) / static_cast<double>(
                            lr.calls);
                        LogInfo("CUDA linkage encode large-root stats (total): calls=" + std::to_string(lr.calls) +
                            " root_chunks_avg=" + FormatFloatLocal(avg_root_chunks, 3) +
                            " root_chunks_max=" + std::to_string(lr.root_chunks_max) +
                            " have_xc0_full_calls=" + std::to_string(lr.have_xc0_full_calls) +
                            " root_chunk_h_min=" + std::to_string(lr.root_chunk_h_min) +
                            " root_chunk_h_max=" + std::to_string(lr.root_chunk_h_max) +
                            " root_icm_fast=" + std::to_string(lr.root_icm_fast_calls) +
                            " root_icm_slow=" + std::to_string(lr.root_icm_slow_calls) +
                            " sample_chunk_calls=" + std::to_string(lr.sample_chunk_calls) +
                            " sample_chunks_total=" + std::to_string(lr.sample_chunks_total) +
                            " sample_chunk_n_min=" + std::to_string(lr.sample_chunk_n_min) +
                            " sample_chunk_n_max=" + std::to_string(lr.sample_chunk_n_max));
                        if (lr.root_icm_slow_calls > 0) {
                            LogWarn("root_icm_slow=" + std::to_string(lr.root_icm_slow_calls) +
                                " > 0: root ICM used the slow fallback (BuildRe0 + tiled GEMM). "
                                "This usually means root_chunk_h < h0 (xC0 buffer not large enough to hold all root codes). "
                                "Consider increasing runtime.cuda_linkage_large_root_xc0_chunk_mb or "
                                "reducing per-batch n (fewer CUDA pool contexts).");
                        }
                    }
                    else {
                        LogInfo(
                            "CUDA linkage encode large-root stats (total): calls=0 (large-root encoder not used; this is expected when h0 is small, e.g. 256).");
                    }

                    const CudaLinkageEncodeHostStats host = GetAndResetCudaLinkageEncodeHostStats();
                    LogInfo("CUDA linkage encode host overhead (total): calls=" + std::to_string(host.calls) +
                        " ensure_s=" + FormatFloatLocal(host.ensure_s, 6) +
                        " precomp_refresh_calls=" + std::to_string(host.precomp_refresh_calls) +
                        " batch_grow_calls=" + std::to_string(host.batch_grow_calls) +
                        " xC_grow_calls=" + std::to_string(host.xC_grow_calls) +
                        " xC_grow_max_elems=" + std::to_string(host.xC_grow_max_elems));

                    const CudaLinkageEvalShapeStats sh = GetAndResetCudaLinkageEvalShapeStats();
                    const double d2h_mib = static_cast<double>(sh.d2h_bytes) / (1024.0 * 1024.0);
                    LogInfo("CUDA linkage candidate-eval shapes (total): calls=" + std::to_string(sh.calls) +
                        " d2h_mib=" + FormatFloatLocal(d2h_mib, 3) +
                        " max_B=" + std::to_string(sh.max_B) +
                        " max_npairs=" + std::to_string(sh.max_npairs));
                    if (sh.calls > 0) {
                        auto hist12 = [](const std::uint64_t h[12]) -> std::string
                        {
                            return std::to_string(h[0]) + "," + std::to_string(h[1]) + "," +
                                std::to_string(h[2]) + "," + std::to_string(h[3]) + "," +
                                std::to_string(h[4]) + "," + std::to_string(h[5]) + "," +
                                std::to_string(h[6]) + "," + std::to_string(h[7]) + "," +
                                std::to_string(h[8]) + "," + std::to_string(h[9]) + "," +
                                std::to_string(h[10]) + "," + std::to_string(h[11]);
                        };
                        LogInfo("CUDA linkage candidate-eval shapes hist (total, bucket12 pow2ish): B=[" +
                            hist12(sh.B_hist) + "] npairs=[" + hist12(sh.npairs_hist) + "]");
                    }
                }
                const CudaLinkageEvalWorkspaceStats ws = GetAndResetCudaLinkageEvalWorkspaceStats();
                if (ws.peak_bytes_per_ctx_est > 0) {
                    const double mib = static_cast<double>(ws.peak_bytes_per_ctx_est) / (1024.0 * 1024.0);
                    LogInfo(
                        "CUDA linkage candidate-eval workspace peak(est, total): per_ctx_mib=" + FormatFloatLocal(
                            mib, 2) +
                        " peak_npairs=" + std::to_string(ws.peak_npairs) +
                        " peak_rfull_cols=" + std::to_string(ws.peak_rfull_cols) +
                        " peak_B=" + std::to_string(ws.peak_B) +
                        " (d=" + std::to_string(ws.peak_d) +
                        " H=" + std::to_string(ws.peak_H) +
                        " m=" + std::to_string(ws.peak_m) +
                        " pool_size=" + std::to_string(std::max(1, cfg.runtime.cuda_pool_size)) + ")");
                }
            }
        }
#endif

        if (base_depth_stats_out) {
            LinkageDepthStats stats;
            if (total_real > 0) {
                stats.linkage_ratio =
                    static_cast<float>(static_cast<double>(total_linkaged) / static_cast<double>(total_real));
                stats.max_depth = max_depth;
                stats.mean_depth = static_cast<float>(depth_sum / static_cast<double>(total_real));
            }
            *base_depth_stats_out = stats;
        }
        if (base_linkage_mse_out) {
            MseStats stats;
            if (total_real > 0) {
                stats.mean = static_cast<float>(mse_sum / static_cast<double>(total_real));
                stats.min = static_cast<float>(std::isfinite(mse_min) ? mse_min : 0.0);
                stats.max = static_cast<float>(mse_max);
            }
            *base_linkage_mse_out = stats;
        }

        if (use_coeff_codec) {
            auto bytes_mb = [](std::uint64_t b) -> std::string
            {
                const double mb = static_cast<double>(b) / (1024.0 * 1024.0);
                std::ostringstream oss;
                oss.setf(std::ios::fixed);
                oss << std::setprecision(3) << mb;
                return oss.str();
            };
            const std::uint64_t header_bytes = coeff_stats.lens_bytes + coeff_stats.scales_bytes;
            const std::uint64_t data_bytes = coeff_stats.payload_bytes;
            const std::uint64_t index_bytes = coeff_stats.payload_offsets_bytes + coeff_stats.payload_sizes_bytes;
            LogInfo("Linkage coeff codec (Huffman+scale) bytes: header(lens+scales)=" +
                std::to_string(header_bytes) + " (" + bytes_mb(header_bytes) + " MiB)" +
                " data(payload)=" + std::to_string(data_bytes) + " (" + bytes_mb(data_bytes) + " MiB)" +
                " index(offs+sizes)=" + std::to_string(index_bytes) + " (" + bytes_mb(index_bytes) + " MiB)");
        }

        return true;
    }

} // namespace stlq
