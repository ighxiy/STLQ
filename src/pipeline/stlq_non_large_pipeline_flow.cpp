#include "stlq/pipeline/stlq_non_large_pipeline_flow.h"

#include "stlq/pipeline/app_utils.h"
#include "stlq/linkage/linkage_builder.h"
#include "stlq/linkage/linkage_builder_virtual.h"
#include "stlq/linkage/linkage_reconstruction.h"
#include "stlq/linkage/linkage_summary.h"
#include "stlq/linkage/virtual_augment.h"
#include "stlq/eval/recall_disk.h"
#include "stlq/io/dataset_io.h"
#include "stlq/io/result_io.h"
#include "stlq/ivf/ivf_scan.h"
#include "stlq/common/logger.h"
#include "stlq/quantizer/reconstruct.h"
#include "stlq/common/timer.h"

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

namespace stlq {

int RunNonLargePipelineFlow(Config& config,
                            const std::filesystem::path& large_root,
                            const std::filesystem::path& large_tmp,
                            KernelProvider* kernels,
                            TrainResult& train_result,
                            EvalMetricMode eval_metric_mode,
                            std::string& error) {
    Precomp pre_root;
    bool pre_root_ready = false;
    auto EnsurePreRoot = [&]() -> bool {
        if (pre_root_ready) {
            return true;
        }
        if (!BuildPrecomp(train_result.C_root, &pre_root)) {
            error = "Failed to build precomp for root codebooks.";
            return false;
        }
        pre_root_ready = true;
        return true;
    };

    Dataset base_dataset;
    const bool report_loaded_mse = (!config.base.encode.enabled && !config.base.linkage.enabled);
    const bool need_base_vectors =
        (config.base.encode.enabled || config.base.linkage.enabled || config.virtual_cfg.enabled || report_loaded_mse);
    if (need_base_vectors) {
        LogInfo("Loading base set: " + config.dataset.name);
        Timer load_timer;
        if (!io::LoadBaseSet(config.dataset, &base_dataset.Xb, &error)) {
            LogError(error);
            return 1;
        }
        LogInfo("Loaded Xb=" + DescribeMatrix(base_dataset.Xb) +
                " in " + load_timer.ReportSeconds("time"));
        ApplyRotationInPlace(train_result.R, &base_dataset.Xb);
    }

    int n_base_real = -1;
    BaseEncoding base;
    float base_beam_mse = -1.0f;
    float base_final_mse = -1.0f;
    VirtualEncoding base_virtual;
    if (config.base.encode.enabled) {
        LogInfo("============== BaseSet Encoding ==============");
        Timer encode_timer;
        if (!EnsurePreRoot()) {
            LogError(error);
            return 1;
        }
        if (!EncodeBase(config, base_dataset, train_result.C_root, pre_root, kernels, &base,
                        &base_beam_mse, &base_final_mse, &error)) {
            LogError(error);
            return 1;
        }
        LogInfo(encode_timer.ReportSeconds("Encoding finished in"));
    } else {
        LogInfo("Base encoding disabled; loading base encoding from HDF5...");
        std::string base_load_date = config.io.load_date;
        std::string base_load_seq = config.io.load_seq;
        if (config.io.base_file_set && !config.io.base_file.empty()) {
            std::error_code ec;
            const std::filesystem::path base_path(config.io.base_file);
            const bool base_path_exists = std::filesystem::exists(base_path, ec);
            if (base_path_exists && !ec) {
                base_load_date = "raw";
                base_load_seq.clear();
            }
        }
        if (!io::LoadBaseResultsHq(config.io.base_file,
                                   base_load_date,
                                   base_load_seq,
                                   &base,
                                   &n_base_real,
                                   config.virtual_cfg.enabled ? &base_virtual : nullptr,
                                   &error)) {
            LogError(error);
            return 1;
        }
        if (report_loaded_mse) {
            ColMajorMatrix<float> R_base = ReconstructAll(train_result.C_root.books, base.B, base.a);
            const float mse = MseStatsFromRecon(base_dataset.Xb, R_base).mean;
            LogInfo("Loaded base MSE: " + FormatFloat(mse, 6));
        }
    }
    if (n_base_real <= 0) {
        n_base_real = base.B.cols;
    }

    Dataset query_dataset;
    ColMajorMatrix<int> query_gt_topk;
    const bool do_eval_base = config.eval.base_enabled;
    const bool do_eval_linkage = config.eval.linkage_enabled;
    const bool need_queries = (do_eval_base || do_eval_linkage);
    if (need_queries) {
        LogInfo("Loading query set: " + config.dataset.name);
        {
            Timer load_timer;
            if (!io::LoadQuerySet(config.dataset, &query_dataset.Xq, &query_dataset.gt, &error)) {
                LogError(error);
                return 1;
            }
            LogInfo("Loaded Xq=" + DescribeMatrix(query_dataset.Xq) +
                    " in " + load_timer.ReportSeconds("time"));
        }
        if (NeedsFullGroundtruth(eval_metric_mode)) {
            if (!io::LoadGroundtruthTopK(config.dataset,
                                         query_dataset.Xq.cols,
                                         std::max(1, config.dataset.k),
                                         &query_gt_topk,
                                         &error)) {
                LogError(error);
                return 1;
            }
        }
        ApplyRotationInPlace(train_result.R, &query_dataset.Xq);
    }

    auto BuildEvalMetricsForIndices = [&](const std::vector<int>& gt_first,
                                          const ColMajorMatrix<int>& indices,
                                          EvalMetricBundle* metrics_out) -> bool {
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
    };

    if (do_eval_base) {
        RecallResult base_recall;
        LogInfo("========= Evaluating base recall =========");
        LogInfo(std::string("Base recall mode: ") +
                (config.eval.base_use_ivf ? "ivf" : "full-scan") +
                (config.eval.base_use_ivf ? (" (nprobe=" + std::to_string(std::max(1, config.eval.base_nprobe)) + ")") : ""));
        Timer base_recall_timer;
        const bool use_ivf = config.eval.base_use_ivf;
        if (!(use_ivf ? EvaluateRecallBaseIvf(config, query_dataset, train_result.C_root, base, &base_recall, &error)
                      : EvaluateRecallBase(config, query_dataset, train_result.C_root, base, &base_recall, &error))) {
            LogWarn(error);
        } else {
            EvalMetricBundle metrics;
            if (BuildEvalMetricsForIndices(query_dataset.gt, base_recall.indices, &metrics)) {
                PrintEvalMetrics(metrics);
            }
        }
        LogInfo(base_recall_timer.ReportSeconds("Base recall finished in"));
    }

    BadClusterKnnCache bad_knn_cache;
    if (config.virtual_cfg.enabled && base_virtual.n_virtual() <= 0) {
        LogInfo("Adding virtual nodes...");
        Timer virtual_timer;
        if (!EnsurePreRoot()) {
            LogError(error);
            return 1;
        }
        if (!AddVirtualNodes(config, train_result.C_root, pre_root,
                             base_dataset.Xb, base, train_result.is_bad_cluster,
                             &base_virtual, &bad_knn_cache, &error)) {
            LogError(error);
            return 1;
        }
        LogInfo(virtual_timer.ReportSeconds("Virtual nodes finished in"));
    }

    if (config.io.save_base) {
        std::string base_path = io::MakeUniqueDatedPath(config.io.base_file, &error);
        if (base_path.empty()) {
            LogError(error);
            return 1;
        }
        LogInfo("Saving base results to: " + base_path);
        Timer save_timer;
        if (!io::SaveBaseResultsHq(base_path, config, base, base_final_mse, n_base_real,
                                   base_beam_mse,
                                   config.virtual_cfg.enabled ? &base_virtual : nullptr,
                                   &error)) {
            LogError(error);
            return 1;
        }
        LogInfo(save_timer.ReportSeconds("Base results saved in"));
    }

    LinkageStructure linkage;
    std::vector<int> parent_loaded;
    std::vector<int> cluster_id_loaded;
    if (config.base.linkage.enabled) {
        LogInfo("Building linkage (parent selection + depth ordering)...");
        Timer linkage_timer;
        ColMajorMatrix<float> R_full_linkage;
        if (config.virtual_cfg.enabled) {
            if (!BuildLinkageTwoCodebookVirtual(config.base.linkage, config.hnsw, config.virtual_cfg,
                                              base_dataset.Xb, &base, base_virtual,
                                              train_result.C_root, train_result.C_one,
                                              train_result.is_bad_cluster,
                                              bad_knn_cache.nlist > 0 ? &bad_knn_cache : nullptr,
                                              &linkage, &R_full_linkage, &error)) {
                LogError(error);
                return 1;
            }
        } else if (!BuildLinkageTwoCodebook(config.base.linkage, config.hnsw,
                                          base_dataset.Xb, &base,
                                          train_result.C_root, train_result.C_one,
                                          &linkage, &R_full_linkage, &error)) {
            LogError(error);
            return 1;
        }
        LogInfo(linkage_timer.ReportSeconds("Linkage build finished in"));
        const LinkageDepthStats linkage_stats = ComputeLinkageDepthStats(linkage, base.B.cols);
        const MseStats linkage_mse_stats = MseStatsFromRecon(base_dataset.Xb, R_full_linkage);
        R_full_linkage = {};

        LogInfo("BaseSet linkage summary:");
        LogInfo("  Linkage Ratio: " + FormatFloat(100.0f * linkage_stats.linkage_ratio, 2) + "%");
        LogInfo("  Max depth: " + std::to_string(linkage_stats.max_depth) +
                " Mean depth: " + FormatFloat(linkage_stats.mean_depth, 2));
        LogInfo("  Linkage Error: Max=" + FormatFloat(linkage_mse_stats.max, 4) +
                ", Min=" + FormatFloat(linkage_mse_stats.min, 4) +
                ", Mean=" + FormatFloat(linkage_mse_stats.mean, 4));

        if (config.io.save_linkage) {
            std::string linkage_path = io::MakeUniqueDatedPath(config.io.linkage_file, &error);
            if (linkage_path.empty()) {
                LogError(error);
                return 1;
            }
            LogInfo("Saving linkage results to: " + linkage_path);
            Timer save_timer;
            const int n_total = n_base_real + (config.virtual_cfg.enabled ? base_virtual.n_virtual() : 0);
            std::vector<int> parent = config.virtual_cfg.enabled
                                          ? BuildGlobalParentVirtual(linkage, n_base_real, base_virtual)
                                          : BuildGlobalParent(linkage, n_total);
            std::vector<int> cluster_id(static_cast<std::size_t>(n_total), 0);
            for (const auto& cluster : linkage.clusters) {
                for (int g : cluster.indices) {
                    cluster_id[static_cast<std::size_t>(g)] = cluster.cluster_id;
                }
            }
            if (config.virtual_cfg.enabled) {
                int g_virtual = n_base_real;
                for (int cid = 0; cid < base_virtual.nlist; ++cid) {
                    const int nvc = base_virtual.n_virtual(cid);
                    for (int j = 0; j < nvc && g_virtual < n_total; ++j) {
                        cluster_id[static_cast<std::size_t>(g_virtual)] = cid;
                        ++g_virtual;
                    }
                }
            }
            if (!io::SaveBaseResultsSTLQ(linkage_path, config, base, parent, &cluster_id,
                                           linkage_mse_stats.mean, n_base_real,
                                           linkage_stats.linkage_ratio, linkage_stats.max_depth, linkage_stats.mean_depth,
                                           &train_result.is_bad_cluster,
                                           config.virtual_cfg.enabled ? &base_virtual : nullptr,
                                           &error)) {
                LogError(error);
                return 1;
            }
            LogInfo(save_timer.ReportSeconds("Linkage results saved in"));
        }
    } else {
        LogInfo("Linkage build disabled; loading linkage encoding from HDF5...");
        std::string linkage_load_date = config.io.load_date;
        std::string linkage_load_seq = config.io.load_seq;
        if (config.io.linkage_file_set && !config.io.linkage_file.empty()) {
            std::error_code ec;
            const std::filesystem::path linkage_path(config.io.linkage_file);
            const bool linkage_path_exists = std::filesystem::exists(linkage_path, ec);
            if (linkage_path_exists && !ec) {
                linkage_load_date = "raw";
                linkage_load_seq.clear();
            }
        }
        if (!io::LoadBaseResultsSTLQ(config.io.linkage_file,
                                       linkage_load_date,
                                       linkage_load_seq,
                                       config.io,
                                       &base,
                                       &parent_loaded,
                                       &cluster_id_loaded,
                                       &n_base_real,
                                       config.virtual_cfg.enabled ? &base_virtual : nullptr,
                                       &error)) {
            LogError(error);
            return 1;
        }
        if (report_loaded_mse &&
            (!config.virtual_cfg.enabled || base_virtual.n_virtual() == 0) &&
            static_cast<int>(parent_loaded.size()) == base.B.cols) {
            std::vector<int> depth = ComputeDepthsFromParent(parent_loaded);
            ColMajorMatrix<float> R_linkage = ComputeLinkagedReconstructionMultiBook(train_result.C_root,
                                                                                  train_result.C_one,
                                                                                  base.B,
                                                                                  base.a,
                                                                                  parent_loaded,
                                                                                  depth);
            const float mse = MseStatsFromRecon(base_dataset.Xb, R_linkage).mean;
            LogInfo("Loaded linkage MSE: " + FormatFloat(mse, 6));
        }
    }

    if (do_eval_linkage) {
        RecallResult linkage_recall;
        LogInfo("========= Evaluating linkage recall =========");
        LogInfo(std::string("Linkage recall mode: ivf (nprobe=") +
                std::to_string(std::max(1, config.eval.linkage_nprobe)) + ")");
        Timer linkage_recall_timer;
        if (config.base.linkage.enabled) {
            const bool has_virtual = config.virtual_cfg.enabled && base_virtual.n_virtual() > 0;
            const bool ok_eval = has_virtual
                                     ? EvaluateRecallLinkageVirtual(config, query_dataset,
                                                                  train_result.C_root, train_result.C_one,
                                                                  base, base_virtual, linkage, n_base_real,
                                                                  &linkage_recall, &error)
                                     : EvaluateRecallLinkage(config, query_dataset,
                                                           train_result.C_root, train_result.C_one,
                                                           base, linkage, n_base_real,
                                                           &linkage_recall, &error);
            if (!ok_eval) {
                LogWarn(error);
            } else {
                EvalMetricBundle metrics;
                if (BuildEvalMetricsForIndices(query_dataset.gt, linkage_recall.indices, &metrics)) {
                    PrintEvalMetrics(metrics);
                }
            }
        } else {
            const bool has_virtual = config.virtual_cfg.enabled && base_virtual.n_virtual() > 0;
            if (has_virtual) {
                if (!cluster_id_loaded.empty() && cluster_id_loaded.size() == parent_loaded.size()) {
                    if (!EvaluateRecallLinkageVirtualFromParentWithClusterId(config, query_dataset,
                                                                           train_result.C_root, train_result.C_one,
                                                                           base, base_virtual,
                                                                           parent_loaded, cluster_id_loaded,
                                                                           n_base_real, &linkage_recall, &error)) {
                        LogWarn(error);
                    } else {
                        EvalMetricBundle metrics;
                        if (BuildEvalMetricsForIndices(query_dataset.gt, linkage_recall.indices, &metrics)) {
                            PrintEvalMetrics(metrics);
                        }
                    }
                } else {
                    LogWarn("Virtual-mode linkage recall from flat parent requires cluster_id in the HDF5 file.");
                }
            } else if (!cluster_id_loaded.empty() && cluster_id_loaded.size() == parent_loaded.size()) {
                if (!EvaluateRecallLinkageFromParentWithClusterId(config, query_dataset, train_result.C_root, train_result.C_one,
                                                                base, parent_loaded, cluster_id_loaded,
                                                                n_base_real, &linkage_recall, &error)) {
                    LogWarn(error);
                } else {
                    EvalMetricBundle metrics;
                    if (BuildEvalMetricsForIndices(query_dataset.gt, linkage_recall.indices, &metrics)) {
                        PrintEvalMetrics(metrics);
                    }
                }
            } else if (!EvaluateRecallLinkageFromParent(config, query_dataset, train_result.C_root, train_result.C_one,
                                                      base, parent_loaded,
                                                      n_base_real, &linkage_recall, &error)) {
                LogWarn(error);
            } else {
                EvalMetricBundle metrics;
                if (BuildEvalMetricsForIndices(query_dataset.gt, linkage_recall.indices, &metrics)) {
                    PrintEvalMetrics(metrics);
                }
            }
        }
        LogInfo(linkage_recall_timer.ReportSeconds("Linkage recall finished in"));
    }
    query_dataset.Xq = {};
    query_dataset.gt.clear();
    base_dataset.Xb = {};

    CleanupEmptyLargeWorkspaceDirs(large_root, large_tmp);
    return 0;
}

}  // namespace stlq
