#include "stlq/pipeline/stlq_large_eval_execute.h"

#include "stlq/pipeline/app_utils.h"
#include "stlq/pipeline/stlq_large_eval_archive.h"
#include "stlq/eval/recall_linkage_disk.h"
#include "stlq/common/logger.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

namespace {

struct ProbeBatchSummary {
    int nprobe = 0;
    int ef_search = 0;
    stlq::EvalMetricBundle metrics;
    float r1 = -1.0f;
    double med_core = 0.0;
    double med_qps = 0.0;
    double avg_core = 0.0;
    double avg_qps = 0.0;
};

float RecallAt1OrNeg(const stlq::EvalMetricBundle& metrics) {
    return stlq::EvalMetricAtOrNeg(metrics.recall_curve, 1);
}

std::string FormatRecallQpsSummaryLine(const ProbeBatchSummary& summary, bool include_ef_search) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4)
        << "nprobe=" << summary.nprobe;
    if (include_ef_search) {
        oss << " ef_search=" << summary.ef_search;
    }
    const int report_k = summary.metrics.result_k;
    const float recall_at_k = stlq::EvalMetricAtOrNeg(summary.metrics.recall_curve, report_k);
    const bool recall_k_is_reported_as_rk =
        (report_k == 1 || report_k == 10 || report_k == 50 || report_k == 100);
    if (report_k > 0 && recall_at_k >= 0.0f && !recall_k_is_reported_as_rk) {
        oss << " recall@" << report_k << "=" << recall_at_k;
    }
    const std::string metric_summary = stlq::FormatEvalMetricSummary(summary.metrics);
    if (!metric_summary.empty()) {
        oss << " " << metric_summary;
    }
    oss << std::setprecision(1)
        << " median_qps_core=" << summary.med_qps
        << " avg_qps_core=" << summary.avg_qps;
    return oss.str();
}

std::vector<const ProbeBatchSummary*> SelectParetoByR1(const std::vector<ProbeBatchSummary>& summaries) {
    constexpr double kQpsEps = 1e-9;
    constexpr float kRecallEps = 1e-7f;
    std::vector<const ProbeBatchSummary*> pareto;
    pareto.reserve(summaries.size());
    for (const auto& candidate : summaries) {
        bool dominated = false;
        for (const auto& other : summaries) {
            if (&candidate == &other) continue;
            const bool qps_not_worse = other.avg_qps + kQpsEps >= candidate.avg_qps;
            const bool recall_not_worse = other.r1 + kRecallEps >= candidate.r1;
            const bool strictly_better =
                other.avg_qps > candidate.avg_qps + kQpsEps ||
                other.r1 > candidate.r1 + kRecallEps;
            if (qps_not_worse && recall_not_worse && strictly_better) {
                dominated = true;
                break;
            }
        }
        if (!dominated) {
            pareto.push_back(&candidate);
        }
    }
    std::sort(pareto.begin(), pareto.end(),
              [](const ProbeBatchSummary* lhs, const ProbeBatchSummary* rhs) {
                  if (lhs->avg_qps != rhs->avg_qps) return lhs->avg_qps > rhs->avg_qps;
                  if (lhs->r1 != rhs->r1) return lhs->r1 > rhs->r1;
                  if (lhs->nprobe != rhs->nprobe) return lhs->nprobe < rhs->nprobe;
                  return lhs->ef_search < rhs->ef_search;
              });
    return pareto;
}

}  // namespace

namespace stlq {

void RunLargeLinkageDiskEvalBatchStage(const Config& config,
                                     const TrainResult& train_result,
                                     EvalMetricMode eval_metric_mode,
                                     const LargeRunState& run_state,
                                     const LargeDiskEvalQueryState& eval_query_state,
                                     io::LinkageListReader& linkage_list,
                                     const app::LinkageLOUDSIndexBitBudget& louds_index_budget,
                                     const LargeLinkageProbeEvalPlan& probe_eval_plan,
                                     bool use_coeff_codec,
                                     const std::string& label,
                                     const std::string& archive_label,
                                     std::string& error) {
    const auto& query_dataset = eval_query_state.query_dataset;
    const double linkage_qt_rotate_sec = eval_query_state.linkage_qt_rotate_sec;
    const bool hnsw_probe_mode = probe_eval_plan.hnsw_probe_mode;
    const bool reuse_provider_across_nprobes = probe_eval_plan.reuse_provider_across_nprobes;
    const std::size_t linkage_eval_combo_count = probe_eval_plan.linkage_eval_combo_count;

    std::shared_ptr<void> shared_provider_cache;
    std::vector<ProbeBatchSummary> probe_summaries;
    probe_summaries.reserve(linkage_eval_combo_count);
    for (const LargeLinkageProbeEvalCase& probe_case : probe_eval_plan.cases) {
            Config cfg_probe = config;
            cfg_probe.eval.linkage_nprobe = probe_case.nprobe;
            cfg_probe.eval.linkage_ivf_hnsw_ef_search = probe_case.ef_search;

            const int warmup = std::max(0, cfg_probe.eval.linkage_warmup);
            const int repeat = std::max(1, cfg_probe.eval.linkage_repeat);
            const bool bench = (warmup > 0 || repeat > 1);

            RecallResult linkage_recall;
            DiskLinkageEvalTiming timing{};
            DiskLinkageEvalSession session{};
            if (reuse_provider_across_nprobes && shared_provider_cache) {
                session.provider_cache = shared_provider_cache;
            }
            app::ApplyLOUDSIndexBudgetToTiming(louds_index_budget, &session);
            timing.qt_rotate_wall_sec = linkage_qt_rotate_sec;
            std::vector<double> core_secs;
            core_secs.reserve(static_cast<std::size_t>(repeat));

            if (linkage_eval_combo_count > 1) {
                std::ostringstream hdr;
                hdr << "===== Linkage disk eval nprobe=" << std::max(1, cfg_probe.eval.linkage_nprobe);
                if (hnsw_probe_mode) {
                    hdr << " ef_search=" << std::max(1, cfg_probe.eval.linkage_ivf_hnsw_ef_search);
                }
                hdr << " =====";
                LogInfo(hdr.str());
            }
            LogInfo("========= Evaluating linkage recall (disk IVF, " + label + ") =========");
            LogInfo(std::string("Linkage recall mode: disk-ivf (nprobe=") +
                    std::to_string(std::max(1, cfg_probe.eval.linkage_nprobe)) + ")" +
                    " norm2_mode=" + cfg_probe.eval.disk_norm2_mode +
                    " probe_mode=" + cfg_probe.eval.linkage_ivf_probe_mode +
                    (cfg_probe.eval.linkage_ivf_probe_mode == "hier2"
                         ? " hier2_top_coarse=" + std::to_string(cfg_probe.eval.linkage_ivf_hier2_top_coarse)
                         : "") +
                    (hnsw_probe_mode
                         ? " hnsw_ef_search=" +
                               std::to_string(std::max(1, cfg_probe.eval.linkage_ivf_hnsw_ef_search))
                         : ""));

            const bool prepared =
                cfg_probe.eval.linkage_parent_louds_native_eval
                    ? PrepareRecallLinkageIvfDiskSessionParentLOUDSNative(
                          cfg_probe, linkage_list, train_result, use_coeff_codec,
                          &timing, &session, &error)
                    : PrepareRecallLinkageIvfDiskSession(
                          cfg_probe, linkage_list, train_result, use_coeff_codec,
                          &timing, &session, &error);
            if (!prepared) {
                LogWarn(error);
                continue;
            }
            if (reuse_provider_across_nprobes && !shared_provider_cache && session.provider_cache) {
                shared_provider_cache = session.provider_cache;
            }

            if (bench && warmup > 0) {
                Config cfg_run = cfg_probe;
                cfg_run.eval.bench_quiet = true;
                for (int i = 0; i < warmup; ++i) {
                    const bool ok = cfg_run.eval.linkage_parent_louds_native_eval
                        ? EvaluateRecallLinkageIvfFromDiskTimedParentLOUDSNative(
                              cfg_run, query_dataset, linkage_list, train_result, use_coeff_codec,
                              &linkage_recall, &timing, &session, &error)
                        : EvaluateRecallLinkageIvfFromDiskTimed(
                              cfg_run, query_dataset, linkage_list, train_result, use_coeff_codec,
                              &linkage_recall, &timing, &session, &error);
                    if (!ok) {
                        LogWarn(error);
                        break;
                    }
                }
            }

            for (int i = 0; i < repeat; ++i) {
                LogInfo("--- " + std::to_string(i + 1) + "/" + std::to_string(repeat) + " ---");
                Config cfg_run = cfg_probe;
                cfg_run.eval.bench_quiet = cfg_probe.eval.bench_quiet;
                const bool ok = cfg_run.eval.linkage_parent_louds_native_eval
                    ? EvaluateRecallLinkageIvfFromDiskTimedParentLOUDSNative(
                          cfg_run, query_dataset, linkage_list, train_result, use_coeff_codec,
                          &linkage_recall, &timing, &session, &error)
                    : EvaluateRecallLinkageIvfFromDiskTimed(
                          cfg_run, query_dataset, linkage_list, train_result, use_coeff_codec,
                          &linkage_recall, &timing, &session, &error);
                if (!ok) {
                    LogWarn(error);
                    break;
                }
                core_secs.push_back(timing.core_wall_sec);

                if (cfg_run.eval.bench_quiet) {
                    const int nq = query_dataset.Xq.cols;
                    const double qps_core =
                        (timing.core_wall_sec > 0.0) ? (static_cast<double>(nq) / timing.core_wall_sec) : 0.0;
                    std::ostringstream oss;
                    oss << "Linkage disk recall QPS(" << archive_label << "): nprobe="
                        << std::max(1, cfg_run.eval.linkage_nprobe);
                    if (hnsw_probe_mode) {
                        oss << " ef_search=" << std::max(1, cfg_run.eval.linkage_ivf_hnsw_ef_search);
                    }
                    oss << " run=" << i
                        << " qps_core=" << qps_core;
                    LogInfo(oss.str());
                }
            }

            if (!core_secs.empty()) {
                EvalMetricBundle metrics;
                if (BuildLargeEvalMetricsForIndices(config,
                                                    eval_metric_mode,
                                                    eval_query_state.query_gt_topk,
                                                    query_dataset.gt,
                                                    linkage_recall.indices,
                                                    &metrics)) {
                    PrintEvalMetrics(metrics);
                }
                if (cfg_probe.eval.linkage_archive_eval_result) {
                    std::string archive_err;
                    if (!ArchiveLargeLinkageDiskEvalResult(run_state.effective_run_root,
                                                           archive_label,
                                                           metrics,
                                                           query_dataset.Xq.cols,
                                                           use_coeff_codec,
                                                           cfg_probe,
                                                           timing,
                                                           session,
                                                           core_secs,
                                                           &archive_err)) {
                        LogWarn(archive_err);
                    }
                }
            }

            if (!core_secs.empty()) {
                const int nq = query_dataset.Xq.cols;
                const double med_core = app::Median(core_secs);
                const double med_qps_core = (med_core > 0.0) ? (static_cast<double>(nq) / med_core) : 0.0;
                double sum_core = 0.0;
                for (double v : core_secs) sum_core += v;
                const double avg_core = sum_core / static_cast<double>(core_secs.size());
                const double avg_qps_core = (avg_core > 0.0) ? (static_cast<double>(nq) / avg_core) : 0.0;
                if (bench) {
                    std::ostringstream oss;
                    if (cfg_probe.eval.bench_quiet) {
                        oss << "Linkage disk recall QPS(" << archive_label << ") summary: nprobe="
                            << std::max(1, cfg_probe.eval.linkage_nprobe);
                        if (hnsw_probe_mode) {
                            oss << " ef_search=" << std::max(1, cfg_probe.eval.linkage_ivf_hnsw_ef_search);
                        }
                        oss << " nq=" << nq
                            << " repeat=" << repeat
                            << " median_qps_core=" << med_qps_core
                            << " avg_qps_core=" << avg_qps_core;
                    } else {
                        oss << "Linkage disk recall bench(" << archive_label << ") summary: nprobe="
                            << std::max(1, cfg_probe.eval.linkage_nprobe);
                        if (hnsw_probe_mode) {
                            oss << " ef_search=" << std::max(1, cfg_probe.eval.linkage_ivf_hnsw_ef_search);
                        }
                        oss << " nq=" << nq
                            << " repeat=" << repeat
                            << " | median_core_sec=" << med_core
                            << " median_qps_core=" << med_qps_core
                            << " | avg_core_sec=" << avg_core
                            << " avg_qps_core=" << avg_qps_core;
                    }
                    LogInfo(oss.str());
                }

                ProbeBatchSummary summary;
                summary.nprobe = std::max(1, cfg_probe.eval.linkage_nprobe);
                summary.ef_search = std::max(1, cfg_probe.eval.linkage_ivf_hnsw_ef_search);
                BuildLargeEvalMetricsForIndices(config,
                                                eval_metric_mode,
                                                eval_query_state.query_gt_topk,
                                                query_dataset.gt,
                                                linkage_recall.indices,
                                                &summary.metrics);
                summary.r1 = RecallAt1OrNeg(summary.metrics);
                summary.med_core = med_core;
                summary.med_qps = med_qps_core;
                summary.avg_core = avg_core;
                summary.avg_qps = avg_qps_core;
                probe_summaries.push_back(std::move(summary));
            }
    }

    if (linkage_eval_combo_count > 1 && !probe_summaries.empty()) {
        LogInfo("===== Linkage disk recall batch summary (" + archive_label + ") =====");
        for (const auto& s : probe_summaries) {
            LogInfo(FormatRecallQpsSummaryLine(s, hnsw_probe_mode));
        }

        const std::vector<const ProbeBatchSummary*> pareto = SelectParetoByR1(probe_summaries);
        if (!pareto.empty()) {
            LogInfo("===== Linkage disk recall Pareto summary (" + archive_label + ", by r@1) =====");
            for (const ProbeBatchSummary* s : pareto) {
                LogInfo(FormatRecallQpsSummaryLine(*s, hnsw_probe_mode));
            }
        }
    }
}

}  // namespace stlq
