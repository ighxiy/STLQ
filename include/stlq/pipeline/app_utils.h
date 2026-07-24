#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "stlq/common/config.h"
#include "stlq/eval/eval_metrics.h"
#include "stlq/eval/recall_linkage_disk.h"
#include "stlq/ivf/ivf_scan.h"
#include "stlq/common/types.h"

namespace stlq {

std::string FormatFloat(float value, int precision);

std::vector<float> ComputeRecallCurve(const std::vector<int>& ground_truth,
                                      const ColMajorMatrix<int>& indices);

void PrintRecallCurve(const std::vector<float>& recall);
void PrintEvalMetrics(const EvalMetricBundle& metrics);

void NormalizeHVec(Config* config);

std::string DescribeMatrix(const ColMajorMatrix<float>& mat);

bool IsIdentityRotation(const ColMajorMatrix<float>& R);

void ApplyRotationInPlace(const ColMajorMatrix<float>& R, ColMajorMatrix<float>* X);

// Applies rotation X <- R * X and returns the wall seconds spent in the GEMM call only.
// Allocation / zero-fill of the output buffer is intentionally excluded so the returned
// time is comparable to qt_build GEMM timings (which time GemmRaw only).
double ApplyRotationInPlaceGemmSeconds(const ColMajorMatrix<float>& R, ColMajorMatrix<float>* X);

void NormalizeIoPrefixWithM(Config* config);

bool PrepareLargeWorkspaceDirs(const Config& config,
                               std::filesystem::path* out_root,
                               std::filesystem::path* tmp_root,
                               std::string* error);

void CleanupEmptyLargeWorkspaceDirs(const std::filesystem::path& out_root,
                                    const std::filesystem::path& tmp_root);

// Optional: delete large training intermediate outputs that can be regenerated.
// Guarded by large.cleanup.enabled.  Never changes training/encode semantics; only deletes files.
void CleanupTrainIntermediates(const Config& config,
                               const std::filesystem::path& exp_root);

// ---- Staged cleanup engine (large.cleanup.*) ----

// Stage 1: after base_list is ready.
// Deletes base_basic payload and/or base_list raw files according to resolved flags.
void CleanupAfterBaseListReady(const Config& config,
                               const std::filesystem::path& base_basic_dir,
                               const std::filesystem::path& base_list_dir);

// Stage 2: after linkage_list (+ optional coeff codec) is ready.
// Deletes base_list, linkage_list .f32 files, and/or parent.u32 according to resolved flags.
void CleanupAfterLinkageListReady(const Config& config,
                                const std::filesystem::path& base_basic_dir,
                                const std::filesystem::path& base_list_dir,
                                const std::filesystem::path& linkage_list_dir);

// ---- Eval result archival ----
// Writes recall curve + timing summary to a .txt file in <run_root>/eval_result/.
// Filename: recall_<label>_0x<hash16>.txt where hash16 is a hex digest of eval-relevant config fields.
// If the file already exists (same config), it is overwritten (same config = same result).
// label: "float" or "int8", used in filename prefix.
// recall: full recall curve from ComputeRecallCurve (length = topk); written as one float per line,
//   k=1..topk, no labels, percent with 2 decimals (CSV-importable).
// nq: number of queries.
// eval_hash: precomputed hash of eval-relevant config (use ComputeEvalArchiveHash).
// timing: optional last-run timing breakdown (may be nullptr).
// repeat_core_secs: all per-repeat core_sec values (empty => no bench stats).
// repeat_qps_vals: all per-repeat qps_core values (same length as repeat_core_secs).
// repeat_wall_secs: all per-repeat wall_sec values (may be empty; if provided, archived alongside core stats).
std::uint64_t ComputeEvalArchiveHash(const Config& cfg, bool use_coeff_codec);
std::uint64_t ComputeEvalMetricsArchiveHash(const Config& cfg, EvalMetricMode mode, bool use_coeff_codec);

bool ArchiveEvalResult(const std::string& run_root,
                       const std::string& label,
                       const std::vector<float>& recall,
                       int nq,
                       std::uint64_t eval_hash,
                       bool use_coeff_codec,
                       const Config& cfg,
                       const DiskLinkageEvalTiming* timing,
                       const DiskLinkageEvalSession* session,
                       const std::vector<double>& repeat_core_secs,
                       const std::vector<double>& repeat_qps_vals,
                       std::string* err);
bool ArchiveEvalMetricsResult(const std::string& run_root,
                              const std::string& label,
                              const EvalMetricBundle& metrics,
                              int nq,
                              std::uint64_t eval_hash,
                              bool use_coeff_codec,
                              const Config& cfg,
                              const DiskLinkageEvalTiming* timing,
                              const DiskLinkageEvalSession* session,
                              const std::vector<double>& repeat_core_secs,
                              const std::vector<double>& repeat_qps_vals,
                              std::string* err);

}  // namespace stlq
