#pragma once

#include <memory>
#include <string>

#include "stlq/common/config.h"
#include "stlq/ivf/ivf_scan.h"

namespace stlq::io {
class LinkageListReader;
}  // namespace stlq::io

namespace stlq {

// ---------------------------------------------------------------------------
// Pure timing results from one eval pass.  Every field ends with _sec.
// ---------------------------------------------------------------------------
struct DiskLinkageEvalTiming {
    double core_wall_sec = 0.0;      // qt_rotate + qt_gemm + probe_sel + scan_topk wall time
    // INPUT: set by caller before calling EvaluateRecallLinkageIvfFromDiskTimed.
    // The impl reads timing->qt_rotate_wall_sec directly; do NOT pass it as a separate argument.
    double qt_rotate_wall_sec = 0.0;
    double qt_gemm_wall_sec = 0.0;
    double qt_coarse_gemm_wall_sec = 0.0;
    double qt_root_small_gemm_wall_sec = 0.0;
    double qt_one_gemm_wall_sec = 0.0;
    double qt_other_wall_sec = 0.0;
    double probe_sel_wall_sec = 0.0;
    double scan_topk_wall_sec = 0.0;
    // Prep (pre-scan) LOUDS-load times. louds_load_mode in the session describes how they
    // were measured.
    double louds_wall_sec = 0.0;         // total wall: lazy=I/O+decode, preload=decode-only bench wall
    double louds_cpu_sec = 0.0;          // CPU-only decode wall (no I/O)
    double huffman_wall_sec = 0.0;       // total wall: lazy=I/O+decode, preload=decode-only bench wall
    double huffman_cpu_sec = 0.0;        // CPU-only decode wall (no I/O)
};

// ---------------------------------------------------------------------------
// Non-timing diagnostics + opaque session cache.  Separated from DiskLinkageEvalTiming
// so that timing fields remain pure measurements.
// ---------------------------------------------------------------------------
struct DiskLinkageEvalSession {
    // One-time LOUDS index budget (extra in-memory bits beyond LOUDS payload), amortized by n_real.
    double louds_index_rank_bits_per_real = 0.0;
    double louds_index_select_bits_per_real = 0.0;
    double louds_index_total_bits_per_real = 0.0;
    int louds_index_effective_enabled = 0;
    int louds_index_select_stride = 0;
    int louds_index_rank_words_per_super_log2 = 0;
    // "lazy"   = accumulated from per-call cache-miss PrepStats
    // "preload" = estimated as (bench_avg/x)*nprobe*nq
    std::string louds_load_mode;

    // Opaque handle to the ClusterProvider (+ its dependencies) shared across repeat eval calls.
    // Set automatically on first call to EvaluateRecallLinkageIvfFromDiskTimed; reused on
    // subsequent calls so that preloading and file-handle setup only happen once.
    // Reset to nullptr to force fresh initialization.
    std::shared_ptr<void> provider_cache;
};

bool EvaluateRecallLinkageIvfFromDiskTimed(const Config& cfg,
                                         const Dataset& query_dataset_inmem,
                                         const io::LinkageListReader& linkage_list,
                                         const TrainResult& train,
                                         bool use_coeff_codec,
                                         RecallResult* out,
                                         DiskLinkageEvalTiming* timing,
                                         DiskLinkageEvalSession* session,
                                         std::string* err);

bool PrepareRecallLinkageIvfDiskSession(const Config& cfg,
                                      const io::LinkageListReader& linkage_list,
                                      const TrainResult& train,
                                      bool use_coeff_codec,
                                      DiskLinkageEvalTiming* timing,
                                      DiskLinkageEvalSession* session,
                                      std::string* err);

bool EvaluateRecallLinkageIvfFromDiskTimedParentLOUDSNative(const Config& cfg,
                                                          const Dataset& query_dataset_inmem,
                                                          const io::LinkageListReader& linkage_list,
                                                          const TrainResult& train,
                                                          bool use_coeff_codec,
                                                          RecallResult* out,
                                                          DiskLinkageEvalTiming* timing,
                                                          DiskLinkageEvalSession* session,
                                                          std::string* err);

bool PrepareRecallLinkageIvfDiskSessionParentLOUDSNative(const Config& cfg,
                                                       const io::LinkageListReader& linkage_list,
                                                       const TrainResult& train,
                                                       bool use_coeff_codec,
                                                       DiskLinkageEvalTiming* timing,
                                                       DiskLinkageEvalSession* session,
                                                       std::string* err);

}  // namespace stlq
