#include "stlq/pipeline/stlq_large_eval_archive.h"

#include "stlq/pipeline/app_utils.h"

namespace stlq {

bool ArchiveLargeLinkageDiskEvalResult(const std::string& run_root,
                                     const std::string& archive_label,
                                     const EvalMetricBundle& metrics,
                                     int nq,
                                     bool use_coeff_codec,
                                     const Config& cfg,
                                     const DiskLinkageEvalTiming& timing,
                                     const DiskLinkageEvalSession& session,
                                     const std::vector<double>& repeat_core_secs,
                                     std::string* error) {
    std::vector<double> qps_vals;
    qps_vals.reserve(repeat_core_secs.size());
    for (double cs : repeat_core_secs) {
        qps_vals.push_back((cs > 0.0) ? (static_cast<double>(nq) / cs) : 0.0);
    }

    if (metrics.mode == EvalMetricMode::kTop1Recall) {
        return ArchiveEvalResult(run_root,
                                 archive_label,
                                 metrics.recall_curve,
                                 nq,
                                 ComputeEvalArchiveHash(cfg, use_coeff_codec),
                                 use_coeff_codec,
                                 cfg,
                                 &timing,
                                 &session,
                                 repeat_core_secs,
                                 qps_vals,
                                 error);
    }

    return ArchiveEvalMetricsResult(run_root,
                                    archive_label,
                                    metrics,
                                    nq,
                                    ComputeEvalMetricsArchiveHash(cfg, metrics.mode, use_coeff_codec),
                                    use_coeff_codec,
                                    cfg,
                                    &timing,
                                    &session,
                                    repeat_core_secs,
                                    qps_vals,
                                    error);
}

}  // namespace stlq
