#include "stlq/pipeline/stlq_large_eval_archive.h"
#include "stlq/pipeline/app_utils.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

namespace {

void Check(bool ok, const char* msg) {
    if (!ok) {
        std::cerr << msg << "\n";
        std::exit(1);
    }
}

std::filesystem::path RecallArchivePath(const std::filesystem::path& root,
                                        const std::string& label,
                                        std::uint64_t hash) {
    std::ostringstream fname;
    fname << "recall_" << label << "_0x"
          << std::hex << std::setfill('0') << std::setw(16) << hash
          << ".txt";
    return root / "eval_result" / fname.str();
}

std::filesystem::path MetricsArchivePath(const std::filesystem::path& root,
                                         const std::string& label,
                                         std::uint64_t hash) {
    std::ostringstream fname;
    fname << "metrics_" << label << "_0x"
          << std::hex << std::setfill('0') << std::setw(16) << hash
          << ".txt";
    return root / "eval_result" / fname.str();
}

}  // namespace

int main() {
    namespace fs = std::filesystem;

    const fs::path root = fs::current_path() / "test_large_eval_archive_tmp";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root);

    stlq::Config cfg;
    cfg.eval.linkage_nprobe = 7;
    cfg.eval.linkage_repeat = 2;
    cfg.eval.metric_gt_topks = {1, 10};
    stlq::DiskLinkageEvalTiming timing;
    timing.core_wall_sec = 0.25;
    stlq::DiskLinkageEvalSession session;
    session.louds_load_mode = "test";

    {
        stlq::EvalMetricBundle metrics;
        metrics.mode = stlq::EvalMetricMode::kTop1Recall;
        metrics.nq = 10;
        metrics.result_k = 2;
        metrics.recall_curve = {0.5f, 0.7f};
        std::string err;
        Check(stlq::ArchiveLargeLinkageDiskEvalResult(root.string(),
                                                      "float",
                                                      metrics,
                                                      10,
                                                      /*use_coeff_codec=*/false,
                                                      cfg,
                                                      timing,
                                                      session,
                                                      std::vector<double>{0.1, 0.2},
                                                      &err),
              "top1 archive should succeed");
        const fs::path expected =
            RecallArchivePath(root, "float", stlq::ComputeEvalArchiveHash(cfg, false));
        Check(fs::exists(expected), "top1 archive file should exist");
    }

    {
        stlq::EvalMetricBundle metrics;
        metrics.mode = stlq::EvalMetricMode::kTopKRecall;
        metrics.nq = 10;
        metrics.result_k = 10;
        metrics.recall_curve = {0.5f, 0.7f};
        metrics.gt_curves.push_back(stlq::EvalGtMetricCurve{
            .gt_k = 1,
            .recall_curve = {0.5f, 0.7f},
        });
        std::string err;
        Check(stlq::ArchiveLargeLinkageDiskEvalResult(root.string(),
                                                      "int8",
                                                      metrics,
                                                      10,
                                                      /*use_coeff_codec=*/true,
                                                      cfg,
                                                      timing,
                                                      session,
                                                      std::vector<double>{0.1, 0.2},
                                                      &err),
              "metrics archive should succeed");
        const fs::path expected =
            MetricsArchivePath(root,
                               "int8",
                               stlq::ComputeEvalMetricsArchiveHash(cfg, metrics.mode, true));
        Check(fs::exists(expected), "metrics archive file should exist");
    }

    fs::remove_all(root, ec);
    std::cout << "test_large_eval_archive: OK\n";
    return 0;
}
