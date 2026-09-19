#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace stlq {

struct DatasetConfig {
    std::string name = "SIFT1M";
    std::string data_root = "../data";
    // Optional per-file overrides (relative to `data_root` unless absolute).
    // When empty, dataset-specific defaults are used (based on `dataset.name`).
    //
    // Notes:
    // - "train" corresponds to the learn/training vectors used by quantizer training.
    // - For datasets without a separate learn file (e.g. DEEP10M), default train==base unless overridden.
    // - Datasets with an explicit learn split (e.g. DEEP1M) keep train/base distinct by default.
    std::string train_path;
    std::string base_path;
    std::string query_path;
    std::string groundtruth_path;
    int groundtruth_add1 = 0;  // 0: false, 1: true
    int ntrain = 100000;
    int nbase = 1000000;
    int nquery = 10000;
    int k = 100;
    bool ntrain_set = false;
    bool nbase_set = false;
    bool nquery_set = false;
    bool k_set = false;
};

struct ModelConfig {
    int m = 5;
    // Root / IVF codebook size shortcut. Used only when `h_vec` is not set
    // explicitly; non-root layers default to 256.
    int h0 = 256;
    std::vector<int> h_vec;
    bool h_vec_set = false;
    // Separate size for the depth>0 one-root codebook (C_one layer 0).
    // C_root[0] is the IVF routing layer (may be large, e.g. 16384/65536).
    // C_one[0] typically stays <=256 to keep code0_one stored as uint8.
    int h0_one = 256;
};

struct AdvancedConfig {
    // High-level convenience override for recall-only runs.
    // When true, main_virtual forces:
    //   train.enabled = false
    //   base.encode.enabled = false
    //   base.linkage.enabled = false
    // Eval toggles remain unchanged.
    bool eval_only = false;
};

struct RuntimeConfig {
    int omp_threads = 0;  // 0: keep OpenMP default / env
    // GPU kernels (stlq_gpu only). Requires building with -DSTLQ_ENABLE_CUDA=ON.
    bool use_cuda = false;
    int cuda_device = 0;
    // CUDA mode:
    // - "strict": maximize reproducibility; TF32 off by default.
    // - "fast": allow TF32 tensor cores for GEMM (still no approximate solvers).
    std::string cuda_mode = "strict";
    // Allow TF32 tensor cores for GEMM on Ampere+ (faster, slightly different numerics).
    bool cuda_allow_tf32 = false;

    // Beam-search determinism (large-root init; affects H_beam=2/4 fast CUDA paths):
    // cuBLAS workspace size (MiB). Affects GEMM algorithm selection and stability.
    // Set to 0 to disable explicit workspace.
    int cuda_cublas_workspace_mb = 0;//128 256

    // GPU-accelerate linkage build (candidate-parent evaluation) in the large/streaming pipelines.
    // Enabled automatically when `use_cuda=true` (no secondary toggle).
    // NOTE: linkage build is OpenMP cluster-parallel; GPU execution must be thread-safe.
    int cuda_pool_size = 4;
    // Init-linkage override: CUDA ctx pool size for the init-linkage stage only (large pipeline).
    // 0 => use `runtime.cuda_pool_size`.
    //
    // Motivation: init-linkage (single-codebook, large-root) can have much higher transient VRAM pressure
    // (e.g. h0_root=65536 + large Npairs). Using a smaller pool can avoid OOM while keeping a larger pool
    // for later stages (iteration/baseset) where the workload is more stable.
    int cuda_pool_size_init_linkage = 0;
    int cuda_linkage_min_kp = 16;
    int cuda_linkage_max_kp = 0;  // 0 => no cap

	    // Stage-2 linkage GPU optimization:
	    // When true, keep per-cluster d_R_full on GPU and gather by candidate parent ids.
	    // This removes per-node CPU packing of Rp (d×Kp) into pinned host buffers.
	    bool cuda_linkage_use_device_rfull = true;

	    // Stage-3 linkage GPU optimization:
	    // Batch "inner candidates" evaluation across many nodes in the same layer.
	    // Same-layer candidates stay serial to preserve commit-order semantics.
	    bool cuda_linkage_batch_inner_enabled = true;
			    int cuda_linkage_batch_nodes = 64;
			    int cuda_linkage_batch_max_pairs = 65536;
			    // Init-linkage override: inner/many-nodes batch max pairs for init-linkage stage only.
			    // 0 => use `runtime.cuda_linkage_batch_max_pairs`.
			    int cuda_linkage_batch_max_pairs_init_linkage = 0;
			    // Stage-3e+: opportunistic batch sizing target (Npairs) for linkage GPU many-nodes evaluation
			    // (especially same-layer dynamic window preflush).
		    //
		    // IMPORTANT: this must NOT be used to skip correctness-critical evaluations (e.g. inner candidates).
		    // It only influences when we "preflush" pending work to reduce GPU call count.
		    int cuda_linkage_batch_min_pairs = 2048;

		    // Async disk IO prefetch for linkage build (large/streaming pipelines).
		    // When enabled, use a dedicated IO thread to prefetch per-cluster spans (ivf ids + raw + codes/coeffs)
		    // into a bounded queue, and let OpenMP worker threads consume and run the compute-heavy pipeline.
		    //
		    // This overlaps disk IO with compute and tends to make disk reads more sequential.
		    // NOTE: enabled only when list-order raw exists (base_list has raw_f32/raw_u8) and no random-IO fallback
		    // readers are in use.
		    bool linkage_async_io = false;
		    // Maximum number of clusters to keep prefetched ahead of compute (queue depth).
		    int linkage_async_io_depth = 6;

		    // Async disk IO prefetch for streaming basic encoding (block-based).
		    // When enabled, use a dedicated IO thread to prefetch upcoming base blocks (raw vectors)
		    // into a bounded queue, and let the main compute thread consume and run the compute-heavy encode.
		    //
		    // This is orthogonal to `linkage_async_io` (cluster-based) and is meant to overlap base-set IO
		    // with GPU/CPU compute when you have spare CPU threads / RAM bandwidth.
		    bool basic_async_io = false;
		    // Maximum number of base blocks to keep prefetched ahead of compute (queue depth).
		    int basic_async_io_depth = 2;
		    // Optional RAM budget (MiB) for prefetched basic blocks. 0 disables budget enforcement.
		    // If set, the effective depth will be clamped so (depth * bytes_per_block) <= budget.
		    int basic_async_io_mb = 0;

		    // Async writer for streaming basic encoding (tile-based).
		    // When enabled (GPU only; non-hybrid path), move the per-tile writer->AppendBlock* calls
		    // to a dedicated writer thread, so GPU compute for the next tile can proceed while the
		    // previous tile is being written to disk.
		    //
		    // This does NOT change semantics: outputs are still written strictly in increasing start_id order.
		    bool basic_async_write = true;
		    // Maximum number of in-flight tiles buffered for the writer thread.
		    int basic_async_write_depth = 4;
		    // Optional RAM budget (MiB) for in-flight basic tiles. 0 disables budget enforcement.
		    // If set, the effective depth will be clamped so (depth * bytes_per_tile) <= budget.
		    int basic_async_write_mb = 0;

		    // PrecompLargeRoot CPU cache: build transpose of G0S (root vs small cross table) to speed up
		    // CPU ICM/ILS when h0 is large (e.g., SIFT1B with h0=6144). Extra RAM ~ sizeof(float)*h0*H_small.
		    bool precomp_large_root_g0s_transpose = true;
		    // Optional cap (MiB) for the transpose allocation. 0 disables the cap.
		    int precomp_large_root_g0s_transpose_max_mb = 0;

		    // Hybrid basic encoding (streaming, block-based): run one GPU lane and one CPU lane in parallel,
		    // while preserving ordered output writes by global id.
		    // Default off to keep existing single-lane behavior.
		    bool basic_hybrid_enable = false;
		    // CPU lane scheduling policy: if >0, send ~1/stride blocks to the CPU lane, others to GPU.
		    // 0 => always GPU (even when hybrid is enabled).
		    int basic_hybrid_cpu_stride = 0;
		    // CPU lane OpenMP threads (used only by the CPU lane). 0 => use runtime.omp_threads / env.
		    int basic_hybrid_cpu_threads = 0;
		    // Maximum number of in-flight blocks (queued + completed but not yet written).
		    // This bounds reorder-buffer memory usage. Must be >= 2 when hybrid is enabled.
		    int basic_hybrid_reorder_depth = 6;
		    // Optional additional RAM budget (MiB) for hybrid in-flight blocks. 0 disables budget enforcement.
		    // If set, the effective depth will be clamped so (depth * bytes_per_block) <= budget.
		    int basic_hybrid_inflight_mb = 0;

		    // Stage-3c linkage GPU optimization (good clusters, same-layer dynamic):
		    // Enable same-layer dynamic window/pending many-nodes batching in good clusters.
		    // <=0 disables window/pending batching; >0 also acts as an opportunistic preflush minimum Npairs
		    // (commit-time forced flush is still allowed to run on GPU even when Npairs < min_pairs).
		    int cuda_linkage_same_dynamic_min_pairs = 1;
	    // Stage-3f(1): good clusters, same-layer dynamic window forced flush.
	    // If enabled, evaluate *tiny* forced flushes on CPU to avoid GPU launch/sync overhead.
	    // This does NOT change semantics; it only changes execution location for tiny forced-flush batches.
	    bool cuda_linkage_same_dynamic_tiny_cpu = false;
	    // Init-linkage only (large-root): allow the same tiny forced-flush deferral even when full-precomp
	    // is unavailable (h0_root > 256). This enables a dedicated fixed-root CPU tail evaluator for
	    // init-linkage legacy_root_only. Off by default.
	    bool cuda_linkage_init_large_root_tiny_cpu = false;
	    // Forced-flush slice total (node,candidate) pairs <= max_pairs => CPU evaluator.
	    // Recommended: 8~32. Non-positive disables (even if `tiny_cpu=true`).
	    int cuda_linkage_same_dynamic_tiny_cpu_max_pairs = 16;
	    // Optional safety cap on nodes per tiny-CPU slice. 0 => no cap.
	    int cuda_linkage_same_dynamic_tiny_cpu_max_nodes = 0;
	    // Same-layer dynamic window size (nodes per block). Larger means fewer blocks (fewer GPU launches),
	    // but more in-block dynamic candidates remain serial.
	    int cuda_linkage_same_dynamic_nodes = 64;
	    // Optional: per-node GPU eval threshold for the remaining window-local dynamic candidates.
	    // 0 => use `cuda_linkage_min_kp`. (Effective min is still clamped to >=4 in code.)
	    int cuda_linkage_same_dynamic_min_kp = 0;
	    // Stage-3c (good clusters): enable incremental window-local dynamic many-nodes batching.
	    // This preserves commit-order semantics; it only batches evaluation of the current dynamic candidate views.
	    bool cuda_linkage_same_dynamic_window_batch = false;
	    // Max total (node,candidate) pairs for a window-local dynamic many-nodes flush. 0 => use `cuda_linkage_batch_max_pairs`.
	    int cuda_linkage_same_dynamic_window_max_pairs = 0;
		    // Optional secondary cap: max nodes per window-local dynamic flush. 0 => use full remaining window.
		    int cuda_linkage_same_dynamic_pending_nodes = 0;

	    // Stage-3e (good clusters, same-layer dynamic window preflush):
	    // Optional dedicated preflush target (Npairs). When >0, this overrides `cuda_linkage_batch_min_pairs`
	    // for the same-layer dynamic window preflush scheduler only.
	    //
	    // Motivation: `cuda_linkage_batch_min_pairs` also affects other batching stages (e.g. inner/same_frozen/dyn-before).
	    // The dynamic window preflush often benefits from a different target (typically larger) to reduce
	    // the number of tiny many-nodes GPU calls and keep the GPU busier.
	    //
	    // 0 => disable opportunistic preflush entirely (forced flush still runs as usual).
	    // <0 => use `cuda_linkage_batch_min_pairs` / `cuda_linkage_same_dynamic_min_pairs` (legacy behavior).
	    int cuda_linkage_same_dynamic_preflush_pairs = 0;

	    // Stage-3e (good clusters, same-layer dynamic window preflush):
	    // Optional preflush trigger on "number of nodes with pending work" (in addition to pairs-based trigger).
	    //
	    // Motivation: when same-layer dynamic is extremely sparse (e.g. ~1 parent per node), pending-pair thresholds
	    // may never be reached even though a many-nodes launch would still be worthwhile (especially with async overlap).
	    //
	    // 0 => use a small default (currently min(8, effective_pending_nodes_cap)).
	    int cuda_linkage_same_dynamic_preflush_nodes = 0;

		    // Stage-3b linkage GPU optimization (bad clusters / virtual-front):
		    // Window-batch frozen candidates (virtual + rank(parent)<window_start), while keeping
		    // window-local dynamic candidates serial to preserve commit-order semantics.
		    bool cuda_linkage_bad_window_batch = true;
		    int cuda_linkage_bad_window_nodes = 64;
		    int cuda_linkage_bad_window_max_npairs = 20000;

		    // Stage-4 (capacity): hard cap for CUDA many-nodes evaluator internal chunking.
		    // 0 => disabled (use full Npairs in one shot).
		    // When >0, the evaluator may split a single many-nodes call into multiple device sub-batches
		    // without changing semantics (still evaluates all pairs).
				    int cuda_linkage_chunk_max_pairs = 0;
				    // Init-linkage override: evaluator internal chunk cap for init-linkage only.
				    // 0 => use `runtime.cuda_linkage_chunk_max_pairs`.
				    int cuda_linkage_chunk_max_pairs_init_linkage = 0;
				    // Optional: global CUDA memory budget hint (MiB) for linkage build (capacity planning).
				    // When `cuda_linkage_chunk_max_pairs<=0`, this budget is used to derive an internal
				    // safe chunk cap for the CUDA many-nodes evaluator to reduce peak device memory
				    // (heuristic; does not change semantics). 0 disables.
				    int cuda_linkage_mem_budget_mb = 0;
				    // Init-linkage override: CUDA memory budget hint (MiB) for init-linkage only.
				    // 0 => use `runtime.cuda_linkage_mem_budget_mb`.
				    int cuda_linkage_mem_budget_mb_init_linkage = 0;
					    // Optional host pinned memory budget (MiB) per CUDA ctx for async linkage eval overlap.
					    // When >0, some linkage stages may stage inputs into pinned host buffers and enqueue GPU work
					    // without immediate stream sync, overlapping CPU candidate-build work with GPU eval.
					    // 0 disables async staging/overlap.
					    int cuda_linkage_eval_async_pinned_mb = 0;
					    // Init-linkage override: async pinned budget (MiB) for init-linkage only.
					    // 0 => use `runtime.cuda_linkage_eval_async_pinned_mb`.
					    int cuda_linkage_eval_async_pinned_mb_init_linkage = 0;
		    // Large-root (split-root) root-selection xC0 chunk cap (MiB).
		    // Larger => fewer tiled GEMM calls for root selection (faster), but higher peak VRAM per CUDA ctx.
		    // This does not change semantics (still exact scan over all root codes).
		    int cuda_linkage_large_root_xc0_chunk_mb = 256;
					    // Init-linkage override: xC0 chunk cap (MiB) for init-linkage only.
					    // 0 => use `runtime.cuda_linkage_large_root_xc0_chunk_mb`.
					    int cuda_linkage_large_root_xc0_chunk_mb_init_linkage = 0;
			    // When true, do not silently fall back to CPU when CUDA ctx pool is exhausted.
			    // (Does not affect tiny-cpu small-pairs routing; only ctx acquisition behavior.)
			    bool cuda_linkage_wait_for_ctx = true;

            // Codebook update (C_one) normal-equation sharding.
            //
            // Motivation: exact joint LS for C_one forms a dense H×H normal equation (H=sum(h_l(one))).
            // A naive cluster-parallel implementation that keeps one H×H matrix per OpenMP thread can
            // consume O(omp_threads * H^2) RAM (explodes for m=10).
            //
            // This knob controls a "sharded" accumulation: we limit the number of parallel accumulation
            // workers to `c_one_update_shards` and keep exactly that many accumulator copies.
            //
            // 0 => auto (pick a conservative value).
            int c_one_update_shards = 0;
            // Init-linkage override: sharded accumulation workers for init-stage C_one update only.
            // 0 => use `runtime.c_one_update_shards`.
            int c_one_update_shards_init_linkage = 0;
			};

inline bool EffectiveCudaAllowTf32(const RuntimeConfig& cfg) {
    if (cfg.cuda_mode == "fast") {
        return true;
    }
    return cfg.cuda_allow_tf32;
}

struct LinkageBuildConfig {
    bool enabled = true;
    // Parent-reference construction used before residual encoding.
    // - structured: radial inner-to-outer STLQ topology (production default).
    // - nn_forest: one in-cluster HNSW nearest-neighbor reference per node,
    //              followed by deterministic cycle/depth cuts.
    // - random_forest: one deterministic uniform in-cluster reference per node,
    //                  followed by the same cycle/depth cuts.
    // - all_roots: no parent references; controlled w/o-linkage ablation that
    //              still uses the ordinary linkage codec/evaluation pipeline.
    // - causal_nn: nearest reference from the structured topology's admissible
    //              inner/same-layer-processed candidate set; no cycle cuts.
    // - causal_random: deterministic uniform reference from that same
    //                  admissible candidate set; no cycle cuts.
    std::string reference_policy = "structured";
    double root_percentile = 0.01;
    int num_layers = 16;
    int max_depth = 40;
    int knn_k = 15;
    int depth_k = 15;
    int icm_round = 1;
    bool use_ils = true;
    int ils_rounds = 4;
    int ils_perturb_layers = 3;
    int seed = 38251450;
};

inline bool IsUnrestrictedReferenceForestPolicy(const std::string& policy) {
    return policy == "nn_forest" || policy == "random_forest" || policy == "all_roots";
}

inline bool IsCausalSingleParentPolicy(const std::string& policy) {
    return policy == "causal_nn" || policy == "causal_random";
}

inline bool IsSupportedReferencePolicy(const std::string& policy) {
    return policy == "structured" ||
           IsUnrestrictedReferenceForestPolicy(policy) ||
           IsCausalSingleParentPolicy(policy);
}

inline bool IsSupportedTrainReferencePolicy(const std::string& policy) {
    // The causal controls intentionally target final/base topology only. Init
    // linkage has a separate batched root-code path and is not part of this
    // parent-selection ablation.
    return policy == "structured" || policy == "nn_forest" || policy == "random_forest";
}

struct TrainConfig {
    bool enabled = true;
    // Large streaming training pipeline has a costly init stage (RVQ init + basic encode + init-linkage).
    // If false, skip init stage and instead resume from an existing train checkpoint (io.train_file + io.load_date/seq),
    // then run only the OPQ/global-R iterations.
    bool init_enabled = true;
    struct CheckpointConfig {
        // If true, periodically write train checkpoints during the global-R iteration loop.
        bool enabled = false;
        // Checkpoint period in R-iterations. 0 disables periodic checkpointing.
        int every_R = 0;
    } ckpt;
    bool use_opq_rotation = true;
    // Control training-stage metric reporting (MSE logs + linkage summary).
    // Turn off to reduce logging and avoid some extra metric computations.
    bool log_metrics = true;
	// Diagnostic only: report linkage summary right after linkage build but BEFORE updating C_one.
	// Default off to avoid extra reconstruction scans.
	bool log_linkage_pre_c1 = false;
	// Profiling-only: exit early after RVQ init completes (skip basic encode / ILS+ICM / linkage build).
	bool exit_after_rvq_init = false;
    // Profiling-only: write an extra checkpoint right after init basic finishes and C_root is updated
    // (i.e. after train_basic/train_ivf/train_list are built, but before init-linkage starts).
    // This checkpoint stores only C_root+R (C_one is not yet trained) and is intended for profiling init-linkage
    // without re-running RVQ/basic on very large datasets.
    bool ckpt_after_init_basic = false;
    // If true, stop right after writing the `ckpt_after_init_basic` checkpoint (skip init-linkage).
    bool exit_after_ckpt_init_basic = false;
	int ils_iters = 8;
	int icm_iters = 4;
	int perturb_k = 3;
    int max_R_iters = 20;
    int kmeans_iters = 40;
    double kmeans_tol = 1e-6;
    std::string kmeans_init = "random";
    int init_samples = 100000;
    double kmeans_initial_weight = 1.0;
    double kmeans_min_weight = 0.1;
    double kmeans_outlier_quantile = 0.85;
    double kmeans_cost_threshold = 0.115;
    double kmeans_annealing_factor = 0.9;
    int kmeans_warmup_iters = 2;
    // Approximate quantile for adaptive-weight thresholds using a fixed-range histogram
    // (cost = 1 - dot in [0,2] for spherical k-means). Larger => more precise, but slightly more work.
    // Used only when kmeans_streaming=true and annealing is enabled.
    int kmeans_quantile_bins = 65536;
    // Streaming+annealing mode:
    // - true (default): no-spill two-pass scan (Pass1 histogram, Pass2 rescan+accumulate)
    // - false: legacy spill path (writes per-sample costs/assignments to disk)
    bool kmeans_anneal_no_spill = true;
    // Adaptive-weight anneal scan mode:
    // - "onepass": 1 scan/iter in adaptive phase (threshold lag by 1 iter; requires device weights)
    // - "twopass": legacy 2 scans/iter (Pass1 histogram, Pass2 rescan+update+accumulate)
    std::string kmeans_anneal_mode = "onepass";
    // Prefer device-resident global weights for adaptive annealing (best-effort; may fall back on OOM).
    bool kmeans_anneal_weights_device = true;
    // Debug/testing only: force-disable device weights even if available (exercise fallback paths).
    bool kmeans_force_disable_device_weights = false;
    // K-means training mode:
    // - false: in-memory (materialize X as a matrix; fastest for small datasets)
    // - true: streaming (out-of-core capable; required for 1e7/1e8+)
    bool kmeans_streaming = false;
    // When using CUDA + streaming k-means, try to cache the full normalized X (X_norm) on device if it fits.
    // This makes k-means iterations much faster for moderate N (e.g., 1e6) but hides true streaming H2D cost.
    // Set to false to benchmark streaming efficiency even when X would fit on GPU.
    bool kmeans_cache_xnorm_device = true;
    // Optional partial device cache for streaming k-means when full X_norm does not fit.
    // If >0, cache up to this many MiB of normalized X_norm on GPU (prefix window of the stream) and reuse it
    // across iterations (avoids repeated disk IO + H2D + normalize for that prefix).
    // Best-effort: if allocation or reader seeking fails, falls back to normal streaming.
    int kmeans_device_cache_mb = 0;
    // When full device caching is not used, optionally create a pinned host mirror of X (float32) to speed up
    // repeated H2D copies during streaming k-means (avoids cudaHostRegister; copies X once).
    bool kmeans_pin_host_x = true;
    // For .bvecs streaming k-means, read blocks as uint8 (no CPU u8->f32 expansion) and
    // normalize on GPU (u8 H2D + fused normalize kernel). Keeps .fvecs as float32.
    bool kmeans_bvecs_use_u8 = true;
    // Optional temp dir used by streaming k-means (e.g., for per-iter assignments).
    // If empty, the implementation should fall back to `large.tmp_dir`.
    std::string kmeans_tmp_dir;

    // ---- kmeans|| seeding (scalable kmeans++), GPU-first ----
    // Used only for `train.kmeans_init="kmeans||"` and only for choosing initial centers.
    // Training iterations (assign/update/anneal) remain unchanged.
    bool kmeansll_gpu_enable = true;
    int kmeansll_rounds = 4;
    double kmeansll_oversample = 16.0;
    int kmeansll_candidate_cap = 0;   // 0 => default (e.g. min(16*K, 1'000'000))
    std::uint64_t kmeansll_seed = 0;  // 0 => derive from rng

    // Optional hierarchical seeding (coarse->fine) for very large K.
    bool kmeansll_hier_enable = false;
    int kmeansll_hier_threshold = 16384;
    int kmeansll_hier_fixed_k1 = 256; // <=0 => auto sqrt(K) (future)

    // ---- Large-K hierarchical assignment (A2) ----
    // When enabled and K >= threshold, allow a 2-level coarse→fine assignment to reduce scoring cost.
    int kmeans_large_k_threshold = 8192;
    bool kmeans_large_k_hier2_enable = true;
    int kmeans_large_k_fixed_k1 = 0;
    int kmeans_large_k_k1_min = 128;
    int kmeans_large_k_k1_max = 1024;
    bool kmeans_large_k_k1_pow2 = true;
    int kmeans_large_k_top_coarse = 1;
    bool kmeans_large_k_hier2_train = true;
    bool kmeans_large_k_hier2_encode = true;
    // ---- Optional RVQ init debugging outputs (Prompt-07) ----
    bool save_init_codes = false;
    std::string init_codes_path;
    bool encode_only_after_layer = false;

	    int seed = 1985326;
    // Init-linkage mode:
    // - "legacy_root_only" (default): build init linkage_list using only C_root (single-codebook init-linkage),
    //   then run an exact-LS update to produce the first C_one.
    // - "fast_init": seed C_one from C_root[0] via k-means mapping (h0_root -> h0_one), then build init linkage_list
    //   using the regular two-codebook linkage builder (C_root + seeded C_one) and update C_one with the same
    //   mechanism as the iteration-stage linkage update.
    // - "hybrid": same legacy C_root-only path, but uses hybrid ILS policy: the first GPU ILS round runs
    //   full VarRoot (layer0 perturbable + root ICM GEMM over h0), subsequent rounds downgrade to ConstRoot
    //   (layer0 frozen, skip root ICM tiled GEMM). This avoids the GEMM explosion for large h0 while
    //   preserving high-quality initial root assignments from the first round.
    //
    // Motivation: large h0_root makes legacy init-linkage expensive; fast_init trades some init quality for speed.
    // Hybrid trades less quality (root ICM only in first round) for significant GPU time savings.
    std::string init_linkage_mode = "legacy_root_only";
    // Init-linkage-specific ICM/ILS overrides.
    // Init-linkage is only a warm start for C_one; lower iteration counts reduce GPU time significantly
    // (each root-layer ILS/ICM round requires a full h0-scan GEMM in the LargeRoot path).
    // -1 = use base.linkage values.
    int init_linkage_icm_round = -1;
    int init_linkage_ils_rounds = -1;
    // Hybrid mode: number of ILS rounds that use VarRoot (full root ICM GEMM).
    // Subsequent rounds use ConstRoot (layer0 frozen, skip root ICM).
    // Only relevant when init_linkage_mode = "hybrid".  Default 1.
    int init_linkage_hybrid_varroot_rounds = 1;
	    LinkageBuildConfig linkage;
	};

struct BaseEncodeConfig {
    bool enabled = true;
    bool use_abs = true;
    int H_beam = 2;
    int ils_iters = 32;
    int icm_iters = 4;
    int perturb_k = 3;
    int hnorms = 256;
    int seed = 38251450;
};

struct BaseConfig {
    BaseEncodeConfig encode;
    LinkageBuildConfig linkage;
};

struct VirtualConfig {
    bool enabled = false;
    // Synthetic-root anchor construction. "subkmeans" is ordinary Euclidean
    // Lloyd k-means with arithmetic centroids; it is deliberately not spherical.
    std::string anchor_policy = "umap";
    int subkmeans_iters = 15;
    double virtual_ratio = 0.10;
    double good_fraction = 0.6;
    int min_virtual = 1;
    int max_virtual = 2000;
    float alpha_bad = 0.3f;
    bool use_fixed_virtual_per_cluster = false;
    int fixed_virtual_per_cluster = 256;

    // Julia reference: demos/umap_like_augment.jl:add_virtual_roots_umap_reencode.
    int umap_knn_k = 50;
    int local_connectivity = 1;
    double overlap_thr = 0.9;
    bool prefer_peaks = true;
    int anchor_neighbor_k = 16;
};

struct HnswConfig {
    int M = 48;
    int candidate_multiplier_good = 1;
    int candidate_multiplier_bad = 1;
    // Optional cap for internal linkage-build HNSW construction ef.
    // <=0 disables the cap and preserves legacy hash/run-tag behavior.
    int ef_construction_cap = 0;
};

struct IOConfig {
    std::string pre_fix = "test";
    // Optional: record the config file path passed via CLI `--config` (for reproducibility/debugging).
    // This is a pure metadata field and must not affect any store hash or training semantics.
    std::string config_file;
    std::string train_file;
    std::string base_file;
    std::string linkage_file;
    bool save_train = false;
    bool save_base = false;
    bool save_linkage = false;
    // Train-result checkpoint format:
    // - "auto": use HDF5 when STLQ_ENABLE_HDF5 is built in, otherwise native bin.
    // - "hdf5": write/read HDF5-compatible .h5 train results.
    // - "bin": write/read native STLQ .stlqbin train results.
    std::string result_format = "auto";
    // HDF5 matrix layout compatibility:
    // - "julia" (default): store matrices with swapped dims (cols,rows) so Julia HDF5.jl reads as (rows,cols)
    // - "cxx": store matrices with dims (rows,cols) as-is (C++ internal native)
    std::string hdf5_layout = "julia";
    // For loading dated outputs: date is YYYYMMDD, seq is XXX (digits, without underscore).
    // If date is empty, default is today's date.
    // If seq is empty, loads the no-seq file (stem_YYYYMMDD.h5).
    // If date is "raw", loads io.*_file exactly (no date/seq suffixing).
    std::string load_date;
    std::string load_seq;

    // Index dtype for reading/writing 1-D index arrays in HDF5 (e.g., parent / cluster_id).
    // - "int": use H5T_NATIVE_INT (default; matches C++ internal ids)
    // - "uint": use H5T_NATIVE_UINT32 (useful for some legacy Julia files)
    std::string index_dtype = "int";

    // Internal flags: if user explicitly sets io.*_file, do not overwrite when io.pre_fix or dataset.name changes.
    bool train_file_set = false;
    bool base_file_set = false;
    bool linkage_file_set = false;
};

struct LargeScaleConfig {
    bool enabled = false;

    // Data format override ("auto" uses file suffix).
    std::string train_format = "auto"; // "fvecs" | "bvecs" | "auto"
    std::string base_format = "auto";
    std::string query_format = "auto";
    std::string gt_format = "auto";    // "ivecs" | "auto"

    // Out-of-core dirs.
    std::string output_dir = "outputs";
    std::string tmp_dir = "tmp";

    // Streaming block sizes (number of vectors).
    int train_block = 200000;
    int base_block = 200000;

    // Base encoding output sharding (number of vectors).
    int base_shard_size = 2000000;

    // Cluster bucketing (avoid many small files).
    int cluster_bucket_size = 256;
    int bucket_flush_mb = 256;

    // Optional extra IO to enable fast per-cluster processing.
    bool write_vector_bucket = true;
    bool write_basic_to_bucket = true;

    // Whether baseset linkage_list stores float coefficients (.f32 files).
    // When false, linkage_list meta sets store_coeffs_f32=0 and .f32 files are never created.
    // Requires large.linkage_coeff_codec.enabled=true so that int8 eval is still possible.
    // Training-stage linkage_list always stores float coeffs regardless of this flag.
    bool base_linkage_store_coeffs_f32 = true;

    // ---- Staged cleanup framework ----
    // Combinable retention/cleanup knobs for disk-constrained workflows.
    // Cleanup keys must NOT affect any store hash or training semantics.
    // NOTE: When cleanup is enabled, we always delete train-stage workspace intermediates
    // (train_basic/train_ivf/train_list/train_linkage_*). This is independent of preset and
    // does not touch the canonical saved train result.
    struct CleanupConfig {
        bool enabled = false;

        // Preset: sets default keep flags. Explicit keep_* flags always override.
        // Presets are designed to be *progressive* (monotonic in deletion):
        //   none < dev_all < eval_both < eval_float_min < eval_int8_min
        // i.e., each later preset deletes everything the previous one would, plus more.
        // This avoids surprising behavior where a “more aggressive” preset deletes less.
        //
        //   "none"           - no deletion (equivalent to enabled=false)
        //   "dev_all"        - delete base_basic payload + base_basic/cluster_id.u32 (keeps IVF CSR lists)
        //   "eval_both"      - dev_all + delete base_basic/ivf_* + delete base_list/ (including raw_u8.bin/raw_f32.bin); keep linkage_list .f32 + coeff codec
        //   "eval_float_min" - eval_both + delete linkage_list/parent.u32 when parent_louds.bin exists (float-only footprint)
        //   "eval_int8_min"  - eval_float_min + delete linkage_list/*.f32 coeff payload (int8-only footprint; requires coeff codec)
        //   "custom"         - no preset defaults; rely entirely on explicit keep_* flags
        std::string preset = "eval_both";

        // Combinable explicit keep flags (-1 = use preset default, 0 = delete, 1 = keep).
        int keep_base_basic = -1;
        // Keep the per-vector root assignment file under base_basic/ (cluster_id.u32).
        // This file can be large (~4GB for 1B vectors). It is needed only to (re)build IVF CSR lists.
        // Safe to delete after ivf_offsets.u64 + ivf_ids.u32 are ready.
        int keep_base_basic_cluster_id = -1;
        // Keep the IVF CSR lists under base_basic/ (ivf_offsets.u64 + ivf_ids.u32).
        // These files are needed for base IVF eval and for some rebuild/update stages.
        // Disk linkage recall uses linkage_list only and does NOT require IVF CSR lists.
        int keep_base_basic_ivf_lists = -1;
        int keep_base_list = -1;
        int keep_base_list_raw = -1;      // raw_u8.bin / raw_f32.bin inside base_list/
        int keep_linkage_list_f32 = -1;     // linkage_list/*.f32 coeff files
        int keep_linkage_parent_u32 = -1;   // linkage_list/parent.u32 when LOUDS exists
    } cleanup;

    // Profiling (debug): print per-stage timings for streaming pipelines.
    bool profile_timing = false;

    // When true, tee all runtime logs (from training start through eval end) into
    // <run_root>/log/run.log for the resolved large-scale run tag.
    bool archive_log = false;

    // Safety switch: when true, never auto-delete and rebuild existing on-disk outputs
    // (even if config/hash matches). This protects expensive results from accidental deletion.
    // You must delete the output directories manually (or change io.pre_fix) to rebuild.
    bool protect_existing_outputs = false;

    // BaseSet linkage_list parent storage:
    // - When false (default), only `parent_louds.bin` is stored and parent pointers can be reconstructed
    //   from LOUDS on-demand for eval/codec rebuild.
    // - When true, also store `linkage_list/parent.u32` (faster for some legacy paths, costs ~4 bytes/node).
    //
    // Training-stage linkage_list outputs may override this to keep parent.u32 for update stages.
    bool linkage_store_parent_u32 = false;

    // Optional: lossy (int8+scale) + lossless (Huffman) coefficient compression for baseset linkage_list.
    // NOTE: training-stage train_linkage_list must keep float coeffs for codebook updates.
    struct LinkageCoeffCodecConfig {
        bool enabled = false;
        // "cluster": 1 Huffman stream per (cluster, group), encoding all layers concatenated.
        // "layer":   1 Huffman stream per (cluster, group, layer).
        std::string granularity = "cluster";
        // bits per layer (signed), size m. If size==1, it will be replicated to m.
        std::vector<int> bits_per_layer{7};
        // Quantile candidates (percent). If empty, defaults follow the Julia prototype.
        std::vector<double> p_first_candidates{99.5, 99.8, 100.0};
        std::vector<double> p_rest_candidates{99.5, 99.8, 100.0};
        bool use_weighted_quantile = false;
        bool allow_clip = true;
        bool fit_scale = true;

        // Optional q refinement (ICM-like), adapted from demos/scale.jl. q is the integer coefficient.
        // 0 disables refinement.
        int q_refine_sweeps = 3;
        // 0 means all layers; otherwise refine only layers [0, q_refine_max_layer).
        int q_refine_max_layer = 0;
        // 0 means unlimited; otherwise constrain each update to |Δq|<=step_limit.
        int q_refine_step_limit = 1;
    } linkage_coeff_codec;

    // Checkpoint/resume for long-running baseset linkage build (cluster-parallel, out-of-order OpenMP).
    // When enabled, the linkage builder writes per-cluster done markers and can resume after interruption
    // without requiring clusters to finish in sequential cid order.
    bool linkage_checkpoint = false;

    // Checkpoint/resume for baseset streaming basic encoding (base_basic store).
    // When enabled, the encoder periodically flushes outputs and writes a checkpoint so a later run can
    // resume from the last committed block without deleting existing outputs.
    // NOTE: This is intended for baseset encoding only; train/iteration stages should keep this off.
    bool base_basic_checkpoint = false;
};

// Resolve effective keep flags from CleanupConfig (preset defaults + explicit overrides).
struct ResolvedCleanupFlags {
    bool keep_base_basic = true;
    bool keep_base_basic_cluster_id = true;
    bool keep_base_basic_ivf_lists = true;
    bool keep_base_list = true;
    bool keep_base_list_raw = true;
    bool keep_linkage_list_f32 = true;
    bool keep_linkage_parent_u32 = true;
};

ResolvedCleanupFlags ResolveCleanupFlags(const LargeScaleConfig& large);

struct EvalConfig {
    // Global switches to enable/disable recall evaluation blocks.
    bool base_enabled = true;
    bool linkage_enabled = true;
    // 0: top1 recall only (legacy behavior)
    // 1: top1 recall + top<g>_r@k' for configured GT top-g sets
    // 2: mode 1 + ndcg<g>@k' (binary relevance on exact top-g membership)
    int metric_mode = 0;
    // GT top-g sets to evaluate when metric_mode>=1. Example: [10, 50].
    // Empty => use a built-in fallback clipped by dataset.k.
    std::vector<int> metric_gt_topks;

    // Base recall evaluation:
    // - use_ivf=false: full-scan over entire base set (legacy linscan)
    // - use_ivf=true: IVF-style scan (pick top-nprobe clusters by query·centroid similarity, then scan only those lists)
    bool base_use_ivf = false;
    int base_nprobe = 8;
    // Disk-IVF eval warmup/repeat (for stable QPS benchmarking).
    // warmup runs are executed but not reported; repeat runs are reported and summarized (median).
    int base_warmup = 0;
    int base_repeat = 1;

    // Linkage recall evaluation (disk IVF + linkage_list):
    // - linkage_use_ivf_disk=false: skip disk linkage recall
    // - linkage_use_ivf_disk=true: IVF-style scan + per-cluster linkaged reconstruction from linkage_list on disk
    bool linkage_use_ivf_disk = false;
    int linkage_nprobe = 8;
    int linkage_query_block = 256;
    int linkage_warmup = 0;
    int linkage_repeat = 1;

    // If true, suppress detailed per-run timing logs inside eval kernels (bench harness prints summaries).
    bool bench_quiet = false;

    // For disk IVF linkage recall, choose which coefficient representation to use.
    // - "auto": if linkage_coeff_codec is present and enabled, run both float and int8 (codec) paths; else float only.
    // - "float": float coeffs only (linkage_list/*.f32).
    // - "int8": int8 coeff codec only (linkage_list/coeff_*.bin + scales).
    // - "both": run float then int8 (codec).
    std::string linkage_coeff_mode = "auto";

    // ---- Parent LOUDS (succinct parent) ----
    // If true and LOUDS blobs exist, prefer decoding parent from LOUDS during recall evaluation.
    // If false, always read parent.u32 (keeps legacy behavior for A/B benchmarking).
    bool linkage_parent_louds_enable = true;
    // Experimental CPU-only eval path that queries parent relationships directly from the
    // per-cluster LOUDS index, without decoding/materializing `parent[]` in eval memory.
    // This path is kept fully parallel to the legacy eval implementation.
    bool linkage_parent_louds_native_eval = false;
    // LOUDS select index sampling stride (every S-th 1). Smaller => faster selects, larger => smaller index.
    int parent_louds_select_stride = 128;
    // Rank directory superblock size in log2(words). Default 4 => 16 words => 1024-bit superblocks.
    // Smaller => smaller rank scan range but larger rank index; larger => smaller rank index but more work in Rank1.
    int parent_louds_rank_words_per_super_log2 = 4;
    // Build Rank/Select indices during LOUDS Deserialize.
    // Default false because current recall path fully decodes parent[] and does not query rank/select.
    bool parent_louds_build_indices = false;
    // Eval path: adaptively store per-cluster parent cache as uint16 when nc<=65535,
    // otherwise keep uint32. Disk format is unchanged. Enabled by default because
    // the eval CPU/GPU pipeline natively supports mixed u16/u32 parent storage.
    bool linkage_parent_adaptive_u16_cache = true;

    // ---- Cluster preload (parallel I/O warm-up before evaluation) ----
    // Load ALL cluster list+coeff data into memory before the first query batch, converting
    // random per-query I/O into a structured parallel bulk load.  This is a large win for
    // large nlist (e.g. SIFT1B nlist=65536) where cold-start I/O across the evaluation loop
    // dominates wall time.
    //   < 0  : disabled (lazy loading; existing behaviour)
    //   = 0  : auto follow_omp — use the process default OMP thread count
    //   > 0  : use exactly N I/O threads
    // NOTE: preload mode also fully loads reusable disk eval caches (float norm2 / norm2 LUT).
    // In lazy mode those caches are read back per-cluster on demand, then kept in the per-cluster
    // cache after first touch.  Ensure sufficient memory before enabling preload on very large datasets.
    int linkage_preload_clusters_io_threads = 0;

    // Number of LOUDS+Huffman decode benchmark iterations to run on the first query block.
    // This measures the realistic OMP-parallel wall time of CPU decode (no I/O latency).
    // Only used in preload mode (linkage_preload_clusters_io_threads >= 0); ignored in lazy mode.
    //
    //   >= 1 : benchmark N OMP-parallel decode iterations on the first query block's x clusters.
    //          Prep-time constant = (avg_wall / x) * linkage_nprobe * nq.
    //          Raw Huffman payloads are freed immediately after the benchmark completes.
    //
    // Default 1; must be >= 1.
    int linkage_louds_huffman_bench_times = 1;

	    // ---- Experimental recall GPU acceleration (eval-only; does not affect on-disk store hashes) ----
	    // Enable CUDA scan experiments for disk IVF linkage recall.
	    // Note: the current recall scan+topk implementation is CPU-only by default; this flag adds a new optional path.
	    bool linkage_gpu_scan_enable = false;
	    // If true, use smaller tiles (TileN=256) inside the GPU scan top-k kernel. This reduces the cost of bitonic sorting
	    // compared to TileN=1024 on many workloads. Default false keeps the legacy TileN=1024 path for A/B testing.
	    bool linkage_gpu_scan_tile256 = false;
    // Limit GPU scan to "small" clusters to keep kernels simple and avoid large temporary buffers. max is 16384 for 4090GPU
    int linkage_gpu_scan_max_nc = 16384;
    // Optional: persist a per-cluster GPU cache across query blocks to reduce repeated host packing / H2D.
    // 0 disables caching (default; preserves current behavior and memory usage).
    int linkage_gpu_scan_cache_mb = 0;

    // Enable CUDA norm2 preparation experiments for disk IVF linkage recall (r_norm2 compute).
    // This is a preload-stage optimization; scan semantics must remain identical.
    bool linkage_gpu_norm_enable = false;

    // ---- Eval IVF probe selection backend ----
    // Controls how cluster selection is performed during linkage disk IVF recall evaluation.
    // - "exact" (default): full GEMM over all h0 root centroids; exact top-nprobe selection.
    // - "hier2": approximate 2-level (coarse→fine) selection via a fixed contiguous grouping.
    //   Coarse K1 ~ sqrt(h0) groups, fine scan within top coarse groups.
    //   Significantly reduces the coarse GEMM cost when h0 is large (e.g., 100k clusters).
    // - "hnsw": approximate selection via a disk-cached HNSW index over unit-normalized root centroids
    //   (inner-product space). Avoids the coarse GEMM entirely.
    std::string linkage_ivf_probe_mode = "exact";  // exact|hier2|hnsw
    // Controls how large-pipeline disk recall paths materialize/use per-sample norm2 terms.
    // Applies to both disk-IVF base recall and disk-IVF linkage recall.
    // - "float" (default): current exact path; keep/use full float norm2 arrays.
    // - "lut": quantize all disk real-node norm2 terms with one global 1D-kmeans LUT.
    // - "lut_sqrt": quantize sqrt(norm2) with a global 1D-kmeans LUT, then square centers back.
    // - "lut_log1p": quantize log(1 + alpha * norm2) with a global 1D-kmeans LUT, then invert centers back.
    // - "lut_tail_weighted": tail-aware global 1D quantizer with heavier weights above p99/p99.9.
    // - "lut_piecewise": split norm2 into bulk/tail segments, allocate bins adaptively, and quantize per segment.
    // - "lut_cluster": legacy per-cluster norm2 LUT path, kept for comparison/debugging.
    std::string disk_norm2_mode = "lut";  // float|lut|lut_sqrt|lut_log1p|lut_tail_weighted|lut_piecewise|lut_cluster
    // Iteration count for the 1D k-means used by eval.disk_norm2_mode="lut", "lut_sqrt",
    // "lut_log1p", "lut_tail_weighted", or "lut_cluster".
    // Controls only the norm2-LUT build step; it does not affect training or base encoding.
    int disk_norm2_lut_kmeans_niter = 25;
    // Alpha used by eval.disk_norm2_mode="lut_log1p".
    double disk_norm2_lut_log_alpha = 1.0;
    // Segment boundaries used by eval.disk_norm2_mode="lut_piecewise".
    // Default: split at p99 and p99.9.
    double disk_norm2_lut_piecewise_p1 = 0.99;
    double disk_norm2_lut_piecewise_p2 = 0.999;
    // Bin allocation score for lut_piecewise:
    // priority = (count_weight * count_share + range_weight * range_share) / (bins+1)
    double disk_norm2_lut_piecewise_count_weight = 0.5;
    double disk_norm2_lut_piecewise_range_weight = 0.5;
    // Number of top coarse groups to scan in hier2 mode (1 or 2). Higher increases recall at the
    // cost of scanning more fine centroids. Only used when linkage_ivf_probe_mode = "hier2".
    int linkage_ivf_hier2_top_coarse = 1;
    // Number of spherical k-means iterations for building hier2 coarse groups (default 25).
    int linkage_ivf_hier2_kmeans_niter = 25;

    // HNSW probe selection knobs (only used when linkage_ivf_probe_mode = "hnsw").
    int linkage_ivf_hnsw_M = 32;
    int linkage_ivf_hnsw_ef_construction = 200;
    int linkage_ivf_hnsw_ef_search = 32;

    // ---- Optional norm2 precomputed cache on disk ----
    // After eval, store per-cluster norm2 cache artifacts for linkage disk recall. In "float" mode this
    // writes linkage_list/norm2_float.f32 or norm2_int8.f32. In "lut" mode this writes per-cluster LUT
    // centers + assignment-code files instead. Caches are reused only when sidecars confirm they
    // match the current linkage_list hash and cover every non-empty cluster. Partial evals therefore do
    // not produce reusable caches for later larger-nprobe runs. When enabled, session preparation
    // proactively builds a full-cache artifact if one is not already present.
    bool linkage_norm2_store_float = false;
    bool linkage_norm2_store_int8 = false;

    // ---- Eval result archival ----
    // After eval completes, save recall curve and timing summary as .txt files in
    // <run_root>/eval_result/. Filename: recall_<float|int8>_0x<hash>.txt where hash encodes
    // eval-relevant config fields (nprobe, probe_mode, omp_threads, query_block, warmup, repeat, etc.).
    // Same config hash => file is overwritten (idempotent). Different config => different file.
    bool linkage_archive_eval_result = false;
};

struct Config {
    DatasetConfig dataset;
    ModelConfig model;
    AdvancedConfig advanced;
    RuntimeConfig runtime;
    TrainConfig train;
    BaseConfig base;
    VirtualConfig virtual_cfg;
    HnswConfig hnsw;
    IOConfig io;
    LargeScaleConfig large;
    EvalConfig eval;
};

Config DefaultConfig(bool virtual_mode);
void ApplyDatasetDefaults(Config* config);

bool LoadConfigFile(const std::string& path, Config* config, std::string* error,
                    std::vector<int>* linkage_nprobe_batch = nullptr,
                    std::vector<int>* linkage_ef_search_batch = nullptr);
void NormalizeModelHVec(Config* config);
bool ApplyOverride(const std::string& key, const std::string& value, Config* config, std::string* error,
                   std::vector<int>* linkage_nprobe_batch = nullptr,
                   std::vector<int>* linkage_ef_search_batch = nullptr);
bool ParseArgs(int argc, char** argv, Config* config, std::string* error, std::string* dump_path,
               std::vector<int>* linkage_nprobe_batch = nullptr,
               std::vector<int>* linkage_ef_search_batch = nullptr);
bool SaveConfigSnapshot(const Config& config, const std::string& path, std::string* error);
std::uint64_t ComputeConfigHash(const Config& config);
// Resume signature: hash of config keys that affect train semantics, excluding iteration-count and checkpoint knobs.
std::uint64_t ComputeTrainResumeSigU64(const Config& config);

}  // namespace stlq
