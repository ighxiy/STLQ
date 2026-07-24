#include "stlq/linkage/linkage_streaming_builders.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
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
#include "stlq/io/base_list_store.h"
#include "stlq/io/bvecs_reader.h"
#include "stlq/io/dataset_reader_factory.h"
#include "stlq/io/linkage_list_store.h"
#include "stlq/io/fbin_reader.h"
#include "stlq/io/fvecs_reader.h"
#include "stlq/io/ivf_lists.h"
#include "stlq/knn/hnsw_cluster_knn.h"
#include "stlq/common/logger.h"
#include "stlq/quantizer/encoder.h"
#include "stlq/quantizer/linear_algebra.h"
#include "stlq/linkage/virtual_augment.h"
#include "stlq/common/timer.h"

#include "stlq/cuda/cuda_stream_kernels_pool.h"
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
#include "stlq/linkage/eval_candidates_cuda.h"
#include "stlq/linkage/linkage_encode_cuda.h"
#endif

namespace stlq {
namespace {
#include "linkage_builder_common.inc"
#include "linkage_builder_two_multicenter.inc"

using linkage::AsyncClusterPrefetcher;
using linkage::ClusterWorkBuf;
using linkage::PrefetchedClusterSpans;
using linkage::BuildPrecompInitLinkageFixedRootTinyCpu;
using linkage::BuildPrecompMetaOnly;
using linkage::ReaderCanReadF32;
using linkage::ReaderCanReadU8;

#include "linkage_cluster_builders.inc"

} // namespace

    bool BuildLinkageOneCodebookVirtualInitStreamingByCluster(const Config& cfg,
                                                              const TrainResult& train,
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
        // See BuildLinkageTwoCodebookVirtualStreamingByCluster() for rationale.
        const int nth = std::max(1, (cfg.runtime.omp_threads > 0) ? cfg.runtime.omp_threads : omp_get_max_threads());
        if (cfg.runtime.omp_threads > 0) {
            omp_set_num_threads(nth);
        }
        const int nlist = ivf_lists.nlist();
        if (nlist <= 0) {
            if (err) *err = "BuildLinkageOneCodebookVirtualInitStreamingByCluster: invalid nlist.";
            return false;
        }

        const int d = base_list.meta().d;
        const int m = cfg.model.m;
        const int m_codes = std::max(0, m - 1);
        if (d <= 0 || m <= 1) {
            if (err) *err = "BuildLinkageOneCodebookVirtualInitStreamingByCluster: invalid d/m.";
            return false;
        }

        // Init-linkage (C_root-only): large pipeline typically avoids full-precomp G(H×H) to prevent
        // catastrophic RAM growth when h0_root is large (e.g. 65536).
        //
        // However, for small h0_root (<=256), full-precomp is cheap and enables:
        // - CPU fallback candidate evaluation (used by "tiny-cpu" forced window flushes)
        // - avoiding some large-root-only restrictions in the init-linkage legacy path
        Precomp pre_root;
        if (cfg.large.enabled) {
            const int h0_root =
                (!train.C_root.books.empty()) ? train.C_root.books.front().cols : 0;
            bool want_cuda_pool_init = false;
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
            want_cuda_pool_init = cfg.runtime.use_cuda &&
                ((cfg.runtime.cuda_pool_size_init_linkage > 0) || (cfg.runtime.cuda_pool_size > 0));
#endif
            if (!want_cuda_pool_init) {
                // CPU-only fallback: we must build full-precomp for C_root (the CPU evaluator requires Precomp::G).
                (void)h0_root;
                if (!BuildPrecomp(train.C_root, &pre_root)) {
                    if (err) {
                        *err =
                            "BuildLinkageOneCodebookVirtualInitStreamingByCluster: failed BuildPrecomp(C_root) for CPU-only init-linkage.";
                    }
                    return false;
                }
            }
            else {
                const bool want_tiny_cpu =
                    cfg.runtime.cuda_linkage_same_dynamic_tiny_cpu &&
                    (cfg.runtime.cuda_linkage_same_dynamic_tiny_cpu_max_pairs > 0);
                const bool allow_full_precomp = (h0_root > 0 && h0_root <= 256);
                if (want_tiny_cpu && allow_full_precomp) {
                    if (!BuildPrecomp(train.C_root, &pre_root)) {
                        if (err)
                            *err =
                                "BuildLinkageOneCodebookVirtualInitStreamingByCluster: failed BuildPrecomp(C_root) for tiny-cpu init-linkage.";
                        return false;
                    }
                }
                else {
                    pre_root = BuildPrecompMetaOnly(train.C_root);
                }
            }
        }
        else {
            if (!BuildPrecomp(train.C_root, &pre_root)) {
                if (err) *err = "BuildLinkageOneCodebookVirtualInitStreamingByCluster: failed BuildPrecomp(C_root).";
                return false;
            }
        }

        const LinkageBuildConfig& linkage_cfg = cfg.base.linkage;
        // IMPORTANT: do NOT mix dtypes. For f32 datasets, never consume raw_u8.bin (legacy clamped output).
        // For u8 datasets, never consume raw_f32.bin.
        const bool has_raw_u8 = base_list.HasRawU8();
        const bool has_raw_f32 = base_list.HasRawF32();

        // Init-linkage can use a separate set of runtime knobs to reduce VRAM pressure (e.g. h0_root=65536),
        // without affecting later iteration/baseset linkage stages.
        RuntimeConfig runtime_init = cfg.runtime;
        const auto pick_pos = [](int init_v, int base_v) -> int { return (init_v > 0) ? init_v : base_v; };
        [[maybe_unused]] const int init_pool_size =
            pick_pos(cfg.runtime.cuda_pool_size_init_linkage, cfg.runtime.cuda_pool_size);
        runtime_init.cuda_linkage_batch_max_pairs =
            pick_pos(cfg.runtime.cuda_linkage_batch_max_pairs_init_linkage, cfg.runtime.cuda_linkage_batch_max_pairs);
        runtime_init.cuda_linkage_chunk_max_pairs =
            pick_pos(cfg.runtime.cuda_linkage_chunk_max_pairs_init_linkage, cfg.runtime.cuda_linkage_chunk_max_pairs);
        runtime_init.cuda_linkage_mem_budget_mb =
            pick_pos(cfg.runtime.cuda_linkage_mem_budget_mb_init_linkage, cfg.runtime.cuda_linkage_mem_budget_mb);
        runtime_init.cuda_linkage_eval_async_pinned_mb =
            pick_pos(cfg.runtime.cuda_linkage_eval_async_pinned_mb_init_linkage,
                     cfg.runtime.cuda_linkage_eval_async_pinned_mb);
        runtime_init.cuda_linkage_large_root_xc0_chunk_mb =
            pick_pos(cfg.runtime.cuda_linkage_large_root_xc0_chunk_mb_init_linkage,
                     cfg.runtime.cuda_linkage_large_root_xc0_chunk_mb);

        // Init-linkage can be profiled the same way as the iteration-stage linkage build.
        // We keep the guard alive for the entire function (it also toggles linkage-encode profiling).
        struct ScopedCudaLinkageEvalProfilingInit
        {
            explicit ScopedCudaLinkageEvalProfilingInit(bool enabled_) : enabled(enabled_) {
#if defined(STLQ_ENABLE_CUDA)
                if (enabled) {
                    SetCudaLinkageEvalProfiling(true);
                }
#endif
            }

            ~ScopedCudaLinkageEvalProfilingInit() {
#if defined(STLQ_ENABLE_CUDA)
                if (enabled) {
                    SetCudaLinkageEvalProfiling(false);
                }
#endif
            }

            bool enabled = false;
        };
        [[maybe_unused]] bool cuda_eval_profile = false;
        [[maybe_unused]] CudaStreamKernelsPool* cuda_pool_ptr = nullptr;

#if defined(STLQ_ENABLE_CUDA)
        std::unique_ptr<CudaStreamKernelsPool> cuda_pool;
        std::unique_ptr<ScopedCudaLinkageEvalProfilingInit> cuda_eval_profile_guard;
        if (cfg.runtime.use_cuda) {
            const int pool_size_cfg = init_pool_size;
            if (pool_size_cfg > 0) {
                const int pool_size = pool_size_cfg;
                CudaPoolConfig pool_cfg;
                pool_cfg.device = cfg.runtime.cuda_device;
                pool_cfg.allow_tf32 = EffectiveCudaAllowTf32(cfg.runtime);
                cuda_pool = std::make_unique<CudaStreamKernelsPool>(pool_size, pool_cfg);
                cuda_pool_ptr = cuda_pool.get();
                {
                    static std::atomic<std::uint64_t> last_tag{0};
                    std::uint64_t tag = 1469598103934665603ull;
                    auto Mix = [&](std::uint64_t v)
                    {
                        tag ^= v + 0x9e3779b97f4a7c15ull + (tag << 6) + (tag >> 2);
                    };
                    Mix(static_cast<std::uint64_t>(pool_cfg.device));
                    Mix(static_cast<std::uint64_t>(pool_cfg.allow_tf32 ? 1 : 0));
                    Mix(static_cast<std::uint64_t>(pool_size));
                    Mix(static_cast<std::uint64_t>(cfg.runtime.cuda_linkage_use_device_rfull ? 1 : 0));
                    Mix(static_cast<std::uint64_t>(cfg.runtime.cuda_linkage_chunk_max_pairs));
                    Mix(static_cast<std::uint64_t>(cfg.runtime.cuda_linkage_mem_budget_mb));
                    Mix(static_cast<std::uint64_t>(cfg.runtime.cuda_linkage_eval_async_pinned_mb));
                    Mix(static_cast<std::uint64_t>(cfg.runtime.cuda_linkage_large_root_xc0_chunk_mb));
                    // Include init-linkage overrides in the tag to keep the header log consistent.
                    Mix(static_cast<std::uint64_t>(cfg.runtime.cuda_pool_size_init_linkage));
                    Mix(static_cast<std::uint64_t>(cfg.runtime.cuda_linkage_batch_max_pairs_init_linkage));
                    Mix(static_cast<std::uint64_t>(cfg.runtime.cuda_linkage_chunk_max_pairs_init_linkage));
                    Mix(static_cast<std::uint64_t>(cfg.runtime.cuda_linkage_mem_budget_mb_init_linkage));
                    Mix(static_cast<std::uint64_t>(cfg.runtime.cuda_linkage_eval_async_pinned_mb_init_linkage));
                    Mix(static_cast<std::uint64_t>(cfg.runtime.cuda_linkage_large_root_xc0_chunk_mb_init_linkage));
                    if (last_tag.exchange(tag, std::memory_order_relaxed) != tag) {
                        LogInfo("CUDA init-linkage: enabled (pool_size=" + std::to_string(pool_size) +
                            ", device_rfull=" + std::to_string(
                                static_cast<int>(cfg.runtime.cuda_linkage_use_device_rfull)) + ")");
                    }
                }
                SetCudaLinkageEvalChunkMaxPairs(runtime_init.cuda_linkage_chunk_max_pairs);
                SetCudaLinkageEvalMemBudgetMb(runtime_init.cuda_linkage_mem_budget_mb);
                SetCudaLinkageEvalAsyncPinnedBudgetMb(runtime_init.cuda_linkage_eval_async_pinned_mb);
                SetCudaLinkageEncodeLargeRootXc0ChunkMb(runtime_init.cuda_linkage_large_root_xc0_chunk_mb);
                SetCudaLinkageEncodeLargeRootWorkspaceMb(runtime_init.cuda_linkage_mem_budget_mb);
            }
            else {
                LogInfo(
                    "CUDA init-linkage: disabled (runtime.cuda_pool_size_init_linkage<=0 and runtime.cuda_pool_size<=0).");
            }
        }
        cuda_eval_profile = cfg.large.profile_timing && (cuda_pool_ptr != nullptr);
        cuda_eval_profile_guard = std::make_unique<ScopedCudaLinkageEvalProfilingInit>(cuda_eval_profile);
#endif
        const bool have_cuda_pool =
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
            (cuda_pool_ptr != nullptr);
#else
                false;
#endif
        if (cfg.large.enabled && !have_cuda_pool) {
            LogWarn(
                "BuildLinkageOneCodebookVirtualInitStreamingByCluster: large.enabled=true but CUDA pool is disabled; "
                "using CPU init-linkage with full-precomp(C_root). This may use more RAM and be slower.");
        }

        // Pass A: precompute per-cluster sizes/prefix sums for the linkage_list store and open a positioned writer.
        io::LinkageListStoreConfig out_cfg = store_cfg;
        out_cfg.nlist = nlist;
        out_cfg.m_codes = m_codes;
        // Init-linkage (C_root-only) still needs to store layer0 codes for depth>0 nodes so that
        // init-stage C_one update can match the C_root-only linkage encoding semantics.
        // Store as u8/u16/u32 depending on h0_root. For h0_root>65536 we must use u32 on disk.
        const int h0_root = (!train.C_root.books.empty()) ? train.C_root.books.front().cols : 0;
        out_cfg.code0_width_bytes = (h0_root > 65536) ? 4 : ((h0_root > 256) ? 2 : 1);
        if (out_cfg.dir.empty()) {
            if (err) *err = "BuildLinkageOneCodebookVirtualInitStreamingByCluster: empty linkage_list dir.";
            return false;
        }
        const bool store_coeffs_f32 = out_cfg.store_coeffs_f32;

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
        plan.n_real.assign(static_cast<std::size_t>(nlist), 0);
        plan.n_virt.assign(static_cast<std::size_t>(nlist), 0);
        plan.depth_len.assign(static_cast<std::size_t>(nlist), fixed_depth_len);

        for (int cid = 0; cid < nlist; ++cid) {
            const std::uint32_t n_real = ivf_lists.ListSize(cid);
            plan.n_real[static_cast<std::size_t>(cid)] = n_real;
            plan.n_virt[static_cast<std::size_t>(cid)] = 0;
            plan.real_offsets[static_cast<std::size_t>(cid) + 1] =
                plan.real_offsets[static_cast<std::size_t>(cid)] + static_cast<std::uint64_t>(n_real);
            plan.virt_offsets[static_cast<std::size_t>(cid) + 1] =
                plan.virt_offsets[static_cast<std::size_t>(cid)];
            plan.depth_offsets_offsets[static_cast<std::size_t>(cid) + 1] =
                plan.depth_offsets_offsets[static_cast<std::size_t>(cid)] + static_cast<std::uint64_t>(fixed_depth_len);
        }

        plan.total_depth_offsets_u32_bytes =
            plan.depth_offsets_offsets.back() * static_cast<std::uint64_t>(sizeof(std::uint32_t));
        plan.total_real_u32_bytes = plan.real_offsets.back() * static_cast<std::uint64_t>(sizeof(std::uint32_t));
        plan.total_parent_u32_bytes = plan.real_offsets.back() * static_cast<std::uint64_t>(sizeof(std::uint32_t));
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
        plan.total_virt_codes_bytes = 0;
        plan.total_virt_coeffs_f32_bytes = 0;
        plan.total_virt_a0_f32_bytes = 0;

        io::LinkageListRandomWriter random_writer;
        if (!random_writer.Open(out_cfg, plan, err)) {
            return false;
        }

        // Pass B: cluster-parallel linkage build + positioned write.
        std::vector<std::uint64_t> cluster_real(static_cast<std::size_t>(nlist), 0);
        std::vector<std::uint64_t> cluster_linkaged(static_cast<std::size_t>(nlist), 0);
        std::vector<double> cluster_depth_sum(static_cast<std::size_t>(nlist), 0.0);
        std::vector<int> cluster_max_depth(static_cast<std::size_t>(nlist), 0);
        std::vector<double> cluster_mse_sum(static_cast<std::size_t>(nlist), 0.0);
        std::vector<double> cluster_mse_min(static_cast<std::size_t>(nlist), std::numeric_limits<double>::infinity());
        std::vector<double> cluster_mse_max(static_cast<std::size_t>(nlist), 0.0);

        struct LinkageStreamingTlsTiming
        {
            double read_ivf = 0.0;
            double read_raw = 0.0;
            double read_base = 0.0;
            double linkage_core = 0.0;
            // Time spent blocked waiting for the async IO prefetch queue (not included in read_*).
            double wait_async_pop = 0.0;
            double write = 0.0;
        };
        std::vector<LinkageStreamingTlsTiming> timing_tls(static_cast<std::size_t>(nth));

        std::atomic<int> done{0};
        std::atomic<long long> last_print_ns{0};
        std::atomic<bool> ok{true};
        std::string first_err;
        std::mutex err_mu;
        const bool profile_timing = cfg.large.profile_timing;

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

        const bool can_async_io =
            cfg.runtime.linkage_async_io && cfg.runtime.linkage_async_io_depth > 0 &&
            (has_raw_f32 || has_raw_u8);
        const double wall_t0_init_linkage = profile_timing ? stlq::linkage::LinkageBuildWallNowS() : 0.0;
        double wall_total_init_linkage = 0.0;
        std::unique_ptr<AsyncClusterPrefetcher> async_io;
        if (can_async_io) {
            async_io = std::make_unique<AsyncClusterPrefetcher>(base_list,
                                                                ivf_lists,
                                                                nlist,
                                                                /*skip_done=*/nullptr,
                                                                has_raw_f32,
                                                                has_raw_u8,
                                                                cfg.runtime.linkage_async_io_depth,
                                                                profile_timing,
                                                                &ok,
                                                                &first_err,
                                                                &err_mu,
                                                                "BuildLinkageOneCodebookVirtualInitStreamingByCluster/async_io");
            async_io->Start();
        }

#pragma omp parallel num_threads(nth) default(none) shared(timing_tls, cpu_kernels_tls, done, last_print_ns, ok, first_err, err_mu, async_io, base_list, ivf_lists, train, cfg, linkage_cfg, runtime_init, pre_root, fallback_reader, random_writer, plan, out_cfg, cluster_real, cluster_linkaged, cluster_depth_sum, cluster_max_depth, cluster_mse_sum, cluster_mse_min, cluster_mse_max, kernels, cuda_pool_ptr) firstprivate(now_ns, progress_stream, d, m, m_codes, nlist, fixed_depth_len, allow_random_fallback, profile_timing, kernels_is_cpu, has_raw_u8, has_raw_f32)
        {
            ClusterWorkBuf buf;
            const int tid = omp_get_thread_num();
            LinkageStreamingTlsTiming& t_time = timing_tls[static_cast<std::size_t>(tid)];
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
                    first_err = msg.empty() ? "BuildLinkageOneCodebookVirtualInitStreamingByCluster: failed." : msg;
                }
            };

            auto process_cluster = [&](int cid, int n_real)
            {
                ColMajorMatrix<float>& Xrot = buf.Xrot;
                ColMajorMatrix<Code>& B_small = buf.B_small;
                ColMajorMatrix<float>& a = buf.a;

                ColMajorMatrix<FullCode>& B_full = buf.B_full;
                // FullCode is u8: any h0_root > 256 needs the sideband (code0_width_bytes >= 2).
                const bool need_root_u32 = (out_cfg.code0_width_bytes > 1);
                if (need_root_u32) {
                    buf.code0_root.assign(static_cast<std::size_t>(n_real), static_cast<RootCode>(cid));
                }
                else {
                    buf.code0_root.clear();
                }
                B_full.rows = m;
                B_full.cols = n_real;
                if (B_full.data.size() < static_cast<std::size_t>(m) * static_cast<std::size_t>(n_real)) {
                    B_full.data.resize(static_cast<std::size_t>(m) * static_cast<std::size_t>(n_real));
                }
                for (int i = 0; i < n_real; ++i) {
                    B_full(0, i) = need_root_u32 ? static_cast<FullCode>(0) : static_cast<FullCode>(cid);
                    for (int l = 1; l < m; ++l) {
                        B_full(l, i) = B_small(l - 1, i);
                    }
                }

                LinkageStructure::Cluster& cluster_out = buf.cluster_out;
                int ls_failures = 0;
                MseStats cluster_linkage_mse{};
                double cluster_mse_sum_local = 0.0;

                cluster_out.cluster_id = cid;
                const double t0_linkage = profile_timing ? now_s() : 0.0;
                if (n_real <= 0) {
                    cluster_out.n_real = 0;
                    cluster_out.n_virtual = 0;
                    cluster_out.indices.clear();
                    cluster_out.parent_local.clear();
                    cluster_out.depth_offsets.assign(2, 0);
                }
                else {
                    try {
                        if (linkage_cfg.use_ils && linkage_cfg.ils_rounds > 0 && linkage_cfg.ils_perturb_layers > 0) {
                            LinkageOneCodebookInnerToOuterOneCluster<true>(linkage_cfg, cfg.hnsw,
                                                                           &runtime_init,
#if defined(STLQ_ENABLE_CUDA)
                                                                           cuda_pool_ptr,
#endif
                                                                           cid,
                                                                           Xrot,
                                                                           train.C_root,
                                                                           pre_root,
                                                                           &B_full,
                                                                           &buf.code0_root,
                                                                           &a,
                                                                           &cluster_out,
                                                                           &ls_failures,
                                                                           &cluster_linkage_mse,
                                                                           &cluster_mse_sum_local);
                        }
                        else {
                            LinkageOneCodebookInnerToOuterOneCluster<false>(linkage_cfg, cfg.hnsw,
                                                                            &runtime_init,
#if defined(STLQ_ENABLE_CUDA)
                                                                            cuda_pool_ptr,
#endif
                                                                            cid,
                                                                            Xrot,
                                                                            train.C_root,
                                                                            pre_root,
                                                                            &B_full,
                                                                            &buf.code0_root,
                                                                            &a,
                                                                            &cluster_out,
                                                                            &ls_failures,
                                                                            &cluster_linkage_mse,
                                                                            &cluster_mse_sum_local);
                        }
                    }
                    catch (const std::exception& e) {
                        fail(std::string("BuildLinkageOneCodebookVirtualInitStreamingByCluster: exception: ") + e.
                            what());
                        return;
                    }
                }
                if (profile_timing) {
                    t_time.linkage_core += now_s() - t0_linkage;
                }

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

                const int n_real_out = cluster_out.n_real;
                const int n_virt_out = 0;
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

                const auto& ids = buf.ids;

                real_ids_depth_order.resize(static_cast<std::size_t>(n_real_out));
                parent_u32.resize(static_cast<std::size_t>(n_real_out));
                depth_offsets_u32.assign(static_cast<std::size_t>(fixed_depth_len),
                                         static_cast<std::uint32_t>(n_real_out));
                if (cluster_out.depth_offsets.size() > static_cast<std::size_t>(fixed_depth_len)) {
                    fail("BuildLinkageOneCodebookVirtualInitStreamingByCluster: depth_offsets overflow.");
                    return;
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
                for (int pos = 0; pos < n_real_out; ++pos) {
                    const auto p =
                        static_cast<std::uint32_t>(cluster_out.parent_local[static_cast<std::size_t>(pos)]);
                    parent_u32[static_cast<std::size_t>(pos)] = p;
                    const int old_local = cluster_out.indices[static_cast<std::size_t>(pos)];
                    real_ids_depth_order[static_cast<std::size_t>(pos)] =
                        (old_local >= 0 && old_local < n_real) ? ids[static_cast<std::size_t>(old_local)] : 0u;
                    a0_depth[static_cast<std::size_t>(pos)] = (old_local >= 0 && old_local < n_real)
                                                                  ? a(0, old_local)
                                                                  : 0.0f;

                    const std::size_t off0 = static_cast<std::size_t>(pos) * code0_bytes;
                    const int code0 =
                        (old_local >= 0 && old_local < n_real)
                            ? (need_root_u32
                                   ? static_cast<int>(buf.code0_root[static_cast<std::size_t>(old_local)])
                                   : static_cast<int>(B_full(0, old_local)))
                            : 0;
                    if (code0_bytes == 1) {
                        code0_one_depth[off0] = static_cast<std::uint8_t>(std::clamp(code0, 0, 255));
                    }
                    else if (code0_bytes == 2) {
                        const std::uint16_t z = static_cast<std::uint16_t>(std::clamp(code0, 0, 65535));
                        std::memcpy(code0_one_depth.data() + off0, &z, sizeof(std::uint16_t));
                    }
                    else {
                        const std::uint32_t z = static_cast<std::uint32_t>(std::max(0, code0));
                        std::memcpy(code0_one_depth.data() + off0, &z, sizeof(std::uint32_t));
                    }

                    if (old_local >= 0 && old_local < n_real) {
                        for (int l = 1; l < m; ++l) {
                            const std::size_t offc =
                                static_cast<std::size_t>(pos) * static_cast<std::size_t>(m_codes) +
                                static_cast<std::size_t>(l - 1);
                            codes_small_depth[offc] = B_full(l, old_local);
                            coeffs_small_depth[offc] = a(l, old_local);
                        }
                    }
                }

                (void)n_virt_out;
                virt_codes_small.resize(0);
                virt_coeffs_small.resize(0);
                virt_a0.resize(0);

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
                    return;
                }
                if (profile_timing) {
                    t_time.write += now_s() - t0_write;
                }

                if (ls_failures > 0) {
                    LogWarn("Cluster " + std::to_string(cid) + " LS failures: " + std::to_string(ls_failures));
                }

                const int cur_done = done.fetch_add(1) + 1;
                if (cur_done < nlist) {
                    const long long now = now_ns();
                    long long last = last_print_ns.load(std::memory_order_relaxed);
                    const bool want_dense = (cur_done <= 16) || ((cur_done % 16) == 0);
                    const bool want_time = (last == 0) || (now - last >= 1000000000LL);
                    if ((want_dense || want_time) &&
                        last_print_ns.compare_exchange_strong(last, now, std::memory_order_relaxed)) {
                        const int shown = done.load(std::memory_order_relaxed);
                        std::fprintf(progress_stream, "\rLinkage init build: %d/%d clusters   ", shown, nlist);
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

            if (use_async_io) {
                PrefetchedClusterSpans item;
                while (ok.load(std::memory_order_relaxed)) {
                    const double t0_pop = profile_timing ? now_s() : 0.0;
                    if (!async_io->Pop(&item)) {
                        break;
                    }
                    if (profile_timing) {
                        t_time.wait_async_pop += now_s() - t0_pop;
                    }
                    local_err.clear();
                    buf.ClearPerCluster();
                    buf.ids = std::move(item.ids);
                    const int cid = item.cid;
                    const int n_real = static_cast<int>(buf.ids.size());

                    ColMajorMatrix<float>& Xrot = buf.Xrot;
                    if (n_real > 0) {
                        Xrot.rows = d;
                        Xrot.cols = n_real;
                        const std::size_t need =
                            static_cast<std::size_t>(d) * static_cast<std::size_t>(n_real);
                        if (Xrot.data.size() < need) {
                            Xrot.data.resize(need);
                        }
                    }
                    else {
                        Xrot.rows = d;
                        Xrot.cols = 0;
                        Xrot.data.clear();
                    }

                    if (n_real > 0) {
                        const double t0_raw = profile_timing ? now_s() : 0.0;
                        if (has_raw_f32) {
                            buf.x_f32 = std::move(item.x_f32);
                            kernels_local->Gemm(false, false, 1.0f, train.R, buf.x_f32, 0.0f, &Xrot);
                        }
                        else if (has_raw_u8) {
                            buf.x_u8 = std::move(item.x_u8);
                            kernels_local->ConvertU8ToF32AndRotate(buf.x_u8, train.R, &Xrot);
                        }
                        else {
                            fail(
                                "BuildLinkageOneCodebookVirtualInitStreamingByCluster: async IO requires list-order raw.");
                            break;
                        }
                        if (profile_timing) {
                            t_time.read_raw += now_s() - t0_raw;
                        }

                        ColMajorMatrix<Code>& B_small = buf.B_small;
                        ColMajorMatrix<float>& a = buf.a;
                        B_small.data = std::move(item.codes);
                        a.data = std::move(item.coeffs);
                        B_small.rows = m_codes;
                        B_small.cols = n_real;
                        a.rows = m;
                        a.cols = n_real;
                        if (static_cast<int>(B_small.data.size()) != m_codes * n_real ||
                            static_cast<int>(a.data.size()) != m * n_real) {
                            fail(
                                "BuildLinkageOneCodebookVirtualInitStreamingByCluster: base_list span size mismatch (async).");
                            break;
                        }
                    }
                    else {
                        ColMajorMatrix<Code>& B_small = buf.B_small;
                        ColMajorMatrix<float>& a = buf.a;
                        B_small.rows = m_codes;
                        B_small.cols = 0;
                        B_small.data.clear();
                        a.rows = m;
                        a.cols = 0;
                        a.data.clear();
                    }

                    process_cluster(cid, n_real);
                }
            }
            else {
#pragma omp for schedule(dynamic, 1)
                for (int cid = 0; cid < nlist; ++cid) {
                    if (!ok.load(std::memory_order_relaxed)) {
                        continue;
                    }
                    local_err.clear();
                    buf.ClearPerCluster();
                    auto& ids = buf.ids;
                    const double t0_ivf = profile_timing ? now_s() : 0.0;
                    if (!ivf_thr.ReadList(ivf_lists, cid, &ids, &local_err)) {
                        fail(local_err);
                        continue;
                    }
                    const int n_real = static_cast<int>(ids.size());
                    const std::uint64_t off = ivf_lists.Offset(cid);
                    if (profile_timing) {
                        t_time.read_ivf += now_s() - t0_ivf;
                    }

                    ColMajorMatrix<float>& Xrot = buf.Xrot;
                    if (n_real > 0) {
                        buf.EnsureReal(d, n_real, m, m_codes);
                    }
                    else {
                        Xrot.rows = d;
                        Xrot.cols = 0;
                        Xrot.data.clear();
                    }

                    const double t0_raw = profile_timing ? now_s() : 0.0;
                    if (n_real > 0) {
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
                                Xrot.data.resize(
                                    static_cast<std::size_t>(Xrot.rows) * static_cast<std::size_t>(Xrot.cols));
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
                                        "BuildLinkageOneCodebookVirtualInitStreamingByCluster: raw vectors missing and random fallback is disabled.");
                                    continue;
                                }
                                fail(
                                    "BuildLinkageOneCodebookVirtualInitStreamingByCluster: no fallback reader provided.");
                                continue;
                            }
                        }
                    }
                    if (profile_timing) {
                        t_time.read_raw += now_s() - t0_raw;
                    }

                    const double t0_base = profile_timing ? now_s() : 0.0;
                    ColMajorMatrix<Code>& B_small = buf.B_small;
                    ColMajorMatrix<float>& a = buf.a;
                    if (n_real > 0) {
                        if (!base_thr.ReadCodesSpan(off, static_cast<std::uint32_t>(n_real), &B_small.data,
                                                    &local_err)) {
                            fail(local_err);
                            continue;
                        }
                        if (!base_thr.ReadCoeffsSpan(off, static_cast<std::uint32_t>(n_real), &a.data, &local_err)) {
                            fail(local_err);
                            continue;
                        }
                        B_small.rows = m_codes;
                        B_small.cols = n_real;
                        a.rows = m;
                        a.cols = n_real;
                        if (static_cast<int>(B_small.data.size()) != m_codes * n_real ||
                            static_cast<int>(a.data.size()) != m * n_real) {
                            fail("BuildLinkageOneCodebookVirtualInitStreamingByCluster: base_list span size mismatch.");
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

                    process_cluster(cid, n_real);
                }
            }
        } // omp parallel

        wall_total_init_linkage =
            profile_timing ? (stlq::linkage::LinkageBuildWallNowS() - wall_t0_init_linkage) : 0.0;

        if (async_io) {
            async_io->Stop();
            if (profile_timing && !timing_tls.empty()) {
                timing_tls[0].read_ivf += async_io->read_ivf_s();
                timing_tls[0].read_raw += async_io->read_raw_s();
                timing_tls[0].read_base += async_io->read_base_s();
            }
        }

        {
            const int final_done = done.load(std::memory_order_relaxed);
            std::fprintf(stderr, "\rLinkage init build: %d/%d clusters   \n", final_done, nlist);
            std::fflush(stderr);
        }

        if (!ok.load()) {
            if (err) {
                *err = first_err.empty() ? "BuildLinkageOneCodebookVirtualInitStreamingByCluster: failed." : first_err;
            }
            return false;
        }
        if (!random_writer.Finish(err)) {
            return false;
        }

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

        // Timing summary is printed only when enabled.
        if (profile_timing) {
            LinkageStreamingTlsTiming sum{};
            for (const auto& t : timing_tls) {
                sum.read_ivf += t.read_ivf;
                sum.read_raw += t.read_raw;
                sum.read_base += t.read_base;
                sum.wait_async_pop += t.wait_async_pop;
                sum.linkage_core += t.linkage_core;
                sum.write += t.write;
            }
            LogInfo("Linkage init build timing(s): read_ivf=" + FormatFloatLocal(sum.read_ivf, 6) +
                " read_raw=" + FormatFloatLocal(sum.read_raw, 6) +
                " read_base=" + FormatFloatLocal(sum.read_base, 6) +
                " wait_async_pop=" + FormatFloatLocal(sum.wait_async_pop, 6) +
                " linkage_core_total=" + FormatFloatLocal(sum.linkage_core, 6) +
                " write=" + FormatFloatLocal(sum.write, 6));
            LogInfo("Linkage init build wall time(s): total=" + FormatFloatLocal(wall_total_init_linkage, 6));

#if defined(STLQ_ENABLE_CUDA)
            // Match the iteration-stage CUDA profiling output so init-linkage can be analyzed directly.
            if (cuda_eval_profile) {
                const CudaLinkageEvalStats stats = GetAndResetCudaLinkageEvalStats();
                LogInfo(
                    "CUDA init-linkage candidate-eval timing (sum over threads, total): calls=" + std::to_string(
                        stats.calls) +
                    " h2d=" + FormatFloatLocal(stats.h2d_s, 6) +
                    " residual=" + FormatFloatLocal(stats.residual_s, 6) +
                    " encode=" + FormatFloatLocal(stats.encode_s, 6) +
                    " reduce=" + FormatFloatLocal(stats.reduce_s, 6) +
                    " d2h=" + FormatFloatLocal(stats.d2h_s, 6) +
                    " total=" + FormatFloatLocal(stats.total_s, 6));

                const CudaLinkageEncodeStats enc = GetAndResetCudaLinkageEncodeStats();
                const CudaLinkageEncodeLargeRootStats lr = GetAndResetCudaLinkageEncodeLargeRootStats();
                LogInfo(
                    "CUDA init-linkage encode breakdown (sum over calls, total): calls=" + std::to_string(enc.calls) +
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
                    LogInfo("CUDA init-linkage encode avg (ms/call, total): total=" + FormatFloatLocal(
                            enc.total_s * inv, 6) +
                        " gemm=" + FormatFloatLocal(enc.gemm_s * inv, 6) +
                        " greedy=" + FormatFloatLocal(enc.greedy_s * inv, 6) +
                        " ls_cost=" + FormatFloatLocal(enc.ls_cost_s * inv, 6) +
                        " icm=" + FormatFloatLocal(enc.icm_s * inv, 6) +
                        " ils_icm=" + FormatFloatLocal(enc.ils_icm_s * inv, 6));
                }
                if (lr.calls > 0) {
                    const double avg_root_chunks =
                        static_cast<double>(lr.root_chunks_total) / static_cast<double>(lr.calls);
                    LogInfo("CUDA init-linkage encode large-root stats (total): calls=" + std::to_string(lr.calls) +
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
                            " > 0: init-linkage root ICM used the slow fallback (BuildRe0 + tiled GEMM). "
                            "This usually means root_chunk_h < h0 (xC0 buffer not large enough to hold all root codes). "
                            "Consider increasing runtime.cuda_linkage_large_root_xc0_chunk_mb_init_linkage or "
                            "reducing per-batch n (fewer CUDA pool contexts).");
                    }
                }
                else {
                    LogInfo(
                        "CUDA init-linkage encode large-root stats (total): calls=0 (large-root encoder not used; this is expected when h0 is small, e.g. 256).");
                }

                const CudaLinkageEncodeHostStats host = GetAndResetCudaLinkageEncodeHostStats();
                LogInfo("CUDA init-linkage encode host overhead (total): calls=" + std::to_string(host.calls) +
                    " ensure_s=" + FormatFloatLocal(host.ensure_s, 6) +
                    " precomp_refresh_calls=" + std::to_string(host.precomp_refresh_calls) +
                    " batch_grow_calls=" + std::to_string(host.batch_grow_calls) +
                    " xC_grow_calls=" + std::to_string(host.xC_grow_calls) +
                    " xC_grow_max_elems=" + std::to_string(host.xC_grow_max_elems));

                const CudaLinkageEvalShapeStats sh = GetAndResetCudaLinkageEvalShapeStats();
                const double d2h_mib = static_cast<double>(sh.d2h_bytes) / (1024.0 * 1024.0);
                LogInfo("CUDA init-linkage candidate-eval shapes (total): calls=" + std::to_string(sh.calls) +
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
                    LogInfo("CUDA init-linkage candidate-eval shapes hist (total, bucket12 pow2ish): B=[" +
                        hist12(sh.B_hist) + "] npairs=[" + hist12(sh.npairs_hist) + "]");
                }

                const CudaLinkageEvalWorkspaceStats ws = GetAndResetCudaLinkageEvalWorkspaceStats();
                if (ws.peak_bytes_per_ctx_est > 0) {
                    const double mib = static_cast<double>(ws.peak_bytes_per_ctx_est) / (1024.0 * 1024.0);
                    LogInfo(
                        "CUDA init-linkage candidate-eval workspace peak(est, total): per_ctx_mib=" + FormatFloatLocal(
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
#endif
        }

        return true;
    }

} // namespace stlq
