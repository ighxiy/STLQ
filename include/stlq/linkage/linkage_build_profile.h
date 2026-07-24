#pragma once

#include <cstdint>
#include <chrono>

namespace stlq {
namespace linkage {

// Thread-local profiling accumulator for large-pipeline linkage build.
// Enabled only when the driver sets the TLS pointer (typically when large.profile_timing=true).
struct LinkageBuildProfileTls {
  // Aggregate wall by cluster type (driver sets this around per-cluster execution).
  double linkage_good_s = 0.0;
  double linkage_bad_s = 0.0;

  // Good-cluster linkage_core breakdown.
  // HNSW build for the current cluster (done once per cluster, before per-node queries).
  double good_hnsw_build_cpu_s = 0.0;
  // Sub-breakdowns of `good_cand_build_cpu_s` (do not sum into the parent bucket again).
  // HNSW query/search time inside `QueryCandidatesHnsw*`.
  double good_hnsw_query_cpu_s = 0.0;
  // Candidate packing/assembly time (splitting to inner/same, appending to batch buffers).
  double good_cand_pack_cpu_s = 0.0;
  double good_cand_build_cpu_s = 0.0;
  double good_gpu_wait_s = 0.0;
  double good_rfull_update_s = 0.0;
  double good_cpu_fallback_s = 0.0;
  double good_commit_cpu_s = 0.0;
  // Stage-3f(1): good same-layer dynamic window forced flush evaluated on CPU (tiny batches).
  double good_dyn_window_tiny_cpu_s = 0.0;
  std::uint64_t good_dyn_window_tiny_cpu_calls = 0;
  std::uint64_t good_dyn_window_tiny_cpu_nodes_sum = 0;
  std::uint64_t good_dyn_window_tiny_cpu_pairs_sum = 0;

  std::uint64_t good_gpu_calls_many_nodes = 0;
  std::uint64_t good_gpu_calls_batch = 0;
  std::uint64_t good_gpu_calls_single = 0;
  std::uint64_t good_rfull_update_calls = 0;

  // Stage-3e profiling-only: many-nodes batch "quality" stats for good clusters.
  // Source order: inner / same_frozen / same_dyn_before / same_dyn_window.
  std::uint64_t good_mn_calls_by_src[4] = {0};
  std::uint64_t good_mn_nodes_sum_by_src[4] = {0};
  std::uint64_t good_mn_pairs_sum_by_src[4] = {0};
  // Batch size histograms (12 buckets, pow2-ish, driver defines the bucket function).
  std::uint64_t good_mn_npairs_hist[12] = {0};
  std::uint64_t good_mn_nodes_hist[12] = {0};
  int good_mn_npairs_max = 0;
  int good_mn_nodes_max = 0;

  // Good-cluster CPU fallback reason counters (per node, best-effort).
  std::uint64_t good_cpu_fallback_nodes = 0;
  // Stage-3f: commit-time forced flush was deemed "tiny", so we intentionally skipped a
  // window-dynamic GPU many-nodes call and completed the node via CPU tail evaluation.
  std::uint64_t good_cpu_fallback_dyn_window_tiny_cpu = 0;
  std::uint64_t good_cpu_fallback_no_cuda = 0;
  std::uint64_t good_cpu_fallback_kp_too_small = 0;
  std::uint64_t good_cpu_fallback_kp_too_large = 0;
  std::uint64_t good_cpu_fallback_device_rfull_disabled = 0;
  std::uint64_t good_cpu_fallback_gpu_error = 0;
  std::uint64_t good_cpu_fallback_same_layer_present = 0;
  std::uint64_t good_cpu_fallback_other = 0;

  // Same-layer dynamic candidate Kp distribution (profiling-only).
  // This is used to judge whether Kp-based thresholds will generalize from SIFT1M to SIFT1B.
  // Bucket scheme lives in the driver (implementation file).
  std::uint64_t good_dyn_kp_hist[12] = {0};
  int good_dyn_kp_max = 0;

  // Bad-cluster pre-stage (subset of bad_knn_umap).
  double bad_knn_s = 0.0;
  double bad_umap_s = 0.0;
  // Stage-3f: bad window-local same-layer dynamic forced flush deemed "tiny"; skip GPU many-nodes and
  // complete via commit-time CPU tail evaluation (subset of `cpu_fallback_s`, do not sum).
  double bad_dyn_window_tiny_cpu_s = 0.0;
  std::uint64_t bad_dyn_window_tiny_cpu_calls = 0;
  std::uint64_t bad_dyn_window_tiny_cpu_nodes_sum = 0;
  std::uint64_t bad_dyn_window_tiny_cpu_pairs_sum = 0;

  // Bad-cluster UMAP sub-breakdown (profiling-only; subset of bad_umap_s).
  double bad_umap_rho_sigma_s = 0.0;
  double bad_umap_build_pdir_s = 0.0;
  double bad_umap_wsym_deg_s = 0.0;
  double bad_umap_seed_select_s = 0.0;
  double bad_umap_barycenter_s = 0.0;
  double bad_umap_encode_s = 0.0;
  double bad_umap_encode_gemm_s = 0.0;
  double bad_umap_encode_greedy_s = 0.0;
  double bad_umap_encode_icm_s = 0.0;
  double bad_umap_encode_recon_s = 0.0;

  // Bad-cluster UMAP "encode" worst single-cluster attribution (for stability analysis).
  // NOTE: these are profiling-only diagnostics; they do not affect control flow.
  double bad_umap_encode_max_cluster_s = 0.0;
  int bad_umap_encode_max_cluster_cid = -1;
  int bad_umap_encode_max_cluster_n_real = 0;
  int bad_umap_encode_max_cluster_k_virtual = 0;
  int bad_umap_encode_max_cluster_n_centers = 0;

  // Linkage-core sub-breakdown (focus on CPU control-flow vs GPU waits vs device R_full maintenance).
  double cand_build_cpu_s = 0.0;
  double gpu_wait_s = 0.0;
  double rfull_update_s = 0.0;
  double cpu_fallback_s = 0.0;
  double commit_cpu_s = 0.0;

  // CPU evaluation inner breakdown (used when dynamic fallback runs on CPU).
  double cpu_quantize_s = 0.0;
  double cpu_ls_s = 0.0;
  double cpu_icm_s = 0.0;
  double cpu_cost_s = 0.0;

  // Counters (for sanity / next-step profiling).
  std::uint64_t gpu_calls_many_nodes = 0;
  std::uint64_t gpu_calls_batch = 0;
  std::uint64_t gpu_calls_single = 0;
  std::uint64_t rfull_update_calls = 0;

  // --- Added later (append-only) ---
  // Keep new fields at the end to reduce the blast radius of incremental rebuild issues when profiling is enabled.
  // Number of device R_full update *batches* issued (1 for per-column updates, >1 for batched scatter flushes).
  std::uint64_t good_rfull_update_batches = 0;
  std::uint64_t rfull_update_batches = 0;

  // Stage-3e/3f(2) diagnostics: split same-layer dynamic window GPU many-nodes calls into
  // opportunistic preflush vs commit-time forced flush.
  //
  // Motivation: `runtime.cuda_linkage_same_dynamic_preflush_pairs` only affects the opportunistic preflush scheduler.
  // If `good_dyn_window_preflush_calls` is near 0, that knob will not move the needle (and dyn_window calls are
  // dominated by forced flushes and/or CPU tails).
  std::uint64_t good_dyn_window_preflush_calls = 0;
  std::uint64_t good_dyn_window_preflush_nodes_sum = 0;
  std::uint64_t good_dyn_window_preflush_pairs_sum = 0;
  std::uint64_t good_dyn_window_forced_calls = 0;
  std::uint64_t good_dyn_window_forced_nodes_sum = 0;
  std::uint64_t good_dyn_window_forced_pairs_sum = 0;

  // Stage-3e diagnostics: split `good_gpu_wait_s` into many-nodes sources + other GPU call kinds.
  // This is intended to answer: "where does good.gpu_wait come from" (inner vs same-frozen vs same-dynamic, etc.).
  //
  // NOTE: these numbers are only meaningful when profiling is enabled; update sites are guarded by
  // `LinkageBuildProfileEnabled()` / TLS being set by the driver.
  double good_gpu_wait_mn_s_by_src[4] = {0.0, 0.0, 0.0, 0.0};
  std::uint64_t good_gpu_wait_mn_calls_by_src[4] = {0, 0, 0, 0};
  double good_gpu_wait_batch_s = 0.0;
  double good_gpu_wait_single_s = 0.0;
};

inline thread_local LinkageBuildProfileTls* g_linkage_build_profile_tls = nullptr;

inline void SetLinkageBuildProfileTls(LinkageBuildProfileTls* tls) {
  g_linkage_build_profile_tls = tls;
}

inline LinkageBuildProfileTls* GetLinkageBuildProfileTls() {
  return g_linkage_build_profile_tls;
}

inline bool LinkageBuildProfileEnabled() {
  return g_linkage_build_profile_tls != nullptr;
}

inline double LinkageBuildWallNowS() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

}  // namespace linkage
}  // namespace stlq
