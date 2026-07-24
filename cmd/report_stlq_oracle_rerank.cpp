#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "stlq/common/config.h"
#include "stlq/common/logger.h"
#include "stlq/eval/eval_metrics.h"
#include "stlq/eval/recall_linkage_disk.h"
#include "stlq/io/dataset_reader_factory.h"
#include "stlq/io/linkage_list_store.h"
#include "stlq/io/result_io.h"
#include "stlq/pipeline/app_utils.h"

namespace {

struct Args {
    std::string run_root;
    std::string out_report;
    int candidate_k = 100;
};

std::string TrimAscii(std::string s) {
    auto is_ws = [](unsigned char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; };
    while (!s.empty() && is_ws(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
    while (!s.empty() && is_ws(static_cast<unsigned char>(s.back()))) s.pop_back();
    return s;
}

std::string StripMatchingQuotes(std::string s) {
    s = TrimAscii(std::move(s));
    if (s.size() >= 2) {
        const char a = s.front();
        const char b = s.back();
        if ((a == '"' && b == '"') || (a == '\'' && b == '\'')) return s.substr(1, s.size() - 2);
    }
    return s;
}

bool ParseArgs(int argc, char** argv, Args* out, std::string* err) {
    if (!out) return false;
    *out = {};
    for (int i = 1; i < argc; ++i) {
        const std::string a(argv[i]);
        auto take = [&](std::string* dst) -> bool {
            if (i + 1 >= argc) return false;
            *dst = argv[++i];
            return true;
        };
        if (a == "--run_root" || a == "--run-root") {
            if (!take(&out->run_root)) {
                if (err) *err = "Missing value for " + a;
                return false;
            }
        } else if (a == "--out" || a == "--report") {
            if (!take(&out->out_report)) {
                if (err) *err = "Missing value for " + a;
                return false;
            }
        } else if (a == "--candidate_k" || a == "--candidate-k") {
            std::string v;
            if (!take(&v)) {
                if (err) *err = "Missing value for " + a;
                return false;
            }
            out->candidate_k = std::max(1, std::atoi(v.c_str()));
        } else if (a == "-h" || a == "--help") {
            std::cout
                << "report_stlq_oracle_rerank\n\n"
                << "Required:\n"
                << "  --run_root <dir>       STLQ run root containing config_snapshot.txt and linkage_list/\n\n"
                << "Optional:\n"
                << "  --candidate_k <int>    STLQ ADC shortlist size to rerank by exact raw distance (default: 100)\n"
                << "  --report <path>        Output report (default: <run_root>/analysis/oracle_rerank_k<K>.txt)\n";
            std::exit(0);
        } else {
            if (err) *err = "Unknown arg: " + a;
            return false;
        }
    }
    if (out->run_root.empty()) {
        if (err) *err = "--run_root is required";
        return false;
    }
    if (out->out_report.empty()) {
        out->out_report = (std::filesystem::path(out->run_root) / "analysis" /
                           ("oracle_rerank_k" + std::to_string(out->candidate_k) + ".txt"))
                              .string();
    }
    return true;
}

std::filesystem::path PickConfigSnapshotPath(const std::filesystem::path& run_root) {
    const std::filesystem::path p1 = run_root / "config_snapshot.txt";
    if (std::filesystem::exists(p1)) return p1;
    const std::filesystem::path p2 = run_root / "config_snapshot.cfg";
    if (std::filesystem::exists(p2)) return p2;
    const std::filesystem::path p3 = run_root / "basic_linux.cfg";
    if (std::filesystem::exists(p3)) return p3;
    return p1;
}

std::string ReadRunStateValue(const std::filesystem::path& run_root, const std::string& key) {
    std::ifstream in(run_root / "run_state.txt");
    if (!in.is_open()) return {};
    const std::string prefix = key + " = ";
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind(prefix, 0) == 0) return StripMatchingQuotes(line.substr(prefix.size()));
    }
    return {};
}

bool PathExistsForLoadMode(const std::string& base_path,
                           const std::string& load_date,
                           const std::string& load_seq) {
    if (base_path.empty()) return false;
    if (load_date == "raw") {
        std::error_code ec;
        return std::filesystem::exists(base_path, ec) && !ec;
    }
    std::string local_err;
    const std::string dated = stlq::io::MakeDatedPathForLoad(base_path, load_date, load_seq, &local_err);
    return !dated.empty();
}

std::string ResolveLoadBasePath(const std::string& base_path,
                                const std::string& load_date,
                                const std::string& load_seq,
                                const std::filesystem::path& run_root,
                                const std::filesystem::path& cfg_path) {
    if (base_path.empty() || PathExistsForLoadMode(base_path, load_date, load_seq)) return base_path;
    const std::filesystem::path p(base_path);
    if (p.is_absolute()) return base_path;
    const std::string stripped = (base_path.rfind("./", 0) == 0) ? base_path.substr(2) : base_path;
    std::vector<std::filesystem::path> roots;
    roots.push_back(std::filesystem::current_path());
    roots.push_back(run_root);
    roots.push_back(cfg_path.parent_path());
    auto append_ancestors = [&](std::filesystem::path cur) {
        while (!cur.empty()) {
            roots.push_back(cur);
            const std::filesystem::path parent = cur.parent_path();
            if (parent == cur) break;
            cur = parent;
        }
    };
    append_ancestors(run_root.parent_path());
    append_ancestors(cfg_path.parent_path());
    for (const auto& root : roots) {
        if (root.empty()) continue;
        const std::string cand = (root / stripped).string();
        if (PathExistsForLoadMode(cand, load_date, load_seq)) return cand;
    }
    return base_path;
}

double SquaredDistance(const float* a, const float* b, int d) {
    double s = 0.0;
    for (int i = 0; i < d; ++i) {
        const double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        s += diff * diff;
    }
    return s;
}

std::vector<float> RecallCurveTop1(const std::vector<int>& gt_first, const stlq::ColMajorMatrix<int>& indices) {
    const int k = indices.rows;
    const int nq = indices.cols;
    std::vector<float> out(static_cast<std::size_t>(k), 0.0f);
    if (k <= 0 || nq <= 0 || static_cast<int>(gt_first.size()) < nq) return out;
    std::vector<int> hits(static_cast<std::size_t>(k), 0);
    for (int q = 0; q < nq; ++q) {
        const int gt = gt_first[static_cast<std::size_t>(q)];
        const int* row = indices.Col(q);
        for (int i = 0; i < k; ++i) {
            if (row[i] == gt) {
                for (int j = i; j < k; ++j) hits[static_cast<std::size_t>(j)] += 1;
                break;
            }
        }
    }
    for (int i = 0; i < k; ++i) out[static_cast<std::size_t>(i)] = static_cast<float>(hits[static_cast<std::size_t>(i)]) / static_cast<float>(nq);
    return out;
}

stlq::ColMajorMatrix<int> OracleRerankByRawDistance(const stlq::Config& cfg,
                                                     const stlq::TrainResult& train,
                                                     const stlq::Dataset& query_dataset,
                                                     const stlq::ColMajorMatrix<int>& adc_indices,
                                                     std::string* err) {
    const int k = adc_indices.rows;
    const int nq = adc_indices.cols;
    const int d = query_dataset.Xq.rows;
    stlq::ColMajorMatrix<int> reranked(k, nq);
    stlq::io::DatasetVectorReader base_reader;
    if (!stlq::io::OpenDatasetVectorReader(cfg, stlq::io::DatasetRole::kBase, &base_reader, err)) return reranked;
    if (base_reader.d != d) {
        if (err) *err = "Base/query dimension mismatch.";
        return reranked;
    }
    for (int q = 0; q < nq; ++q) {
        std::vector<std::uint32_t> ids;
        ids.reserve(static_cast<std::size_t>(k));
        for (int i = 0; i < k; ++i) {
            const int id = adc_indices(i, q);
            if (id >= 0) ids.push_back(static_cast<std::uint32_t>(id));
        }
        stlq::ColMajorMatrix<float> Xb;
        if (!stlq::io::ReadDatasetVectorByIdsF32(base_reader, ids, &Xb, err)) return reranked;
        if (!stlq::IsIdentityRotation(train.R)) stlq::ApplyRotationInPlace(train.R, &Xb);
        std::vector<std::pair<double, int>> scored;
        scored.reserve(ids.size());
        const float* qv = query_dataset.Xq.Col(q);
        for (int i = 0; i < Xb.cols; ++i) {
            scored.push_back({SquaredDistance(qv, Xb.Col(i), d), static_cast<int>(ids[static_cast<std::size_t>(i)])});
        }
        std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) {
            if (a.first != b.first) return a.first < b.first;
            return a.second < b.second;
        });
        int out_i = 0;
        for (; out_i < static_cast<int>(scored.size()) && out_i < k; ++out_i) {
            reranked(out_i, q) = scored[static_cast<std::size_t>(out_i)].second;
        }
        for (; out_i < k; ++out_i) reranked(out_i, q) = -1;
    }
    return reranked;
}

void PrintCurveLine(std::ostringstream* oss, const char* label, const std::vector<float>& curve, int k) {
    if (!oss || curve.empty()) return;
    const int kk = std::max(1, std::min(k, static_cast<int>(curve.size())));
    *oss << label << "@1=" << curve[0];
    for (int x : {2, 5, 10, 20, 50, 100}) {
        if (x <= kk) *oss << " " << label << "@" << x << "=" << curve[static_cast<std::size_t>(x - 1)];
    }
    *oss << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    using namespace stlq;
    Args args;
    std::string err;
    if (!ParseArgs(argc, argv, &args, &err)) {
        LogError(err);
        return 1;
    }

    const std::filesystem::path run_root(args.run_root);
    const std::filesystem::path cfg_path = PickConfigSnapshotPath(run_root);
    Config cfg = DefaultConfig(true);
    NormalizeHVec(&cfg);
    std::vector<int> ignored;
    if (!LoadConfigFile(cfg_path.string(), &cfg, &err, &ignored)) {
        LogError(err);
        return 1;
    }
    cfg.dataset.k = std::max(1, args.candidate_k);
    cfg.dataset.k_set = true;
    cfg.eval.linkage_enabled = true;
    cfg.eval.linkage_use_ivf_disk = true;
    cfg.eval.linkage_repeat = 1;
    cfg.eval.linkage_warmup = 0;
    cfg.eval.linkage_coeff_mode = "int8";
    cfg.eval.bench_quiet = true;

    TrainResult train;
    {
        std::string base_path = cfg.io.train_file;
        std::string load_date = cfg.io.load_date;
        std::string load_seq = cfg.io.load_seq;
        const std::string effective_train_h5 = ReadRunStateValue(run_root, "effective_train_h5");
        if (!effective_train_h5.empty() && effective_train_h5 != "<none>") {
            base_path = effective_train_h5;
            load_date = "raw";
            load_seq.clear();
        }
        base_path = ResolveLoadBasePath(base_path, load_date, load_seq, run_root, cfg_path);
        if (!io::LoadTrainResults(base_path, load_date, load_seq, cfg.model.m, &train, &err)) {
            LogError(err.empty() ? "LoadTrainResults failed." : err);
            return 1;
        }
    }

    Dataset query_dataset;
    if (!io::LoadQuerySet(cfg, &query_dataset.Xq, &query_dataset.gt, &err)) {
        LogError(err);
        return 1;
    }
    if (!IsIdentityRotation(train.R)) ApplyRotationInPlace(train.R, &query_dataset.Xq);

    io::LinkageListReader linkage_list;
    if (!linkage_list.Open((run_root / "linkage_list").string(), &err)) {
        LogError(err);
        return 1;
    }

    RecallResult adc;
    DiskLinkageEvalTiming timing{};
    DiskLinkageEvalSession session{};
    timing.qt_rotate_wall_sec = 0.0;
    const bool use_coeff_codec = true;
    bool prepared = cfg.eval.linkage_parent_louds_native_eval
                        ? PrepareRecallLinkageIvfDiskSessionParentLOUDSNative(cfg, linkage_list, train, use_coeff_codec, &timing, &session, &err)
                        : PrepareRecallLinkageIvfDiskSession(cfg, linkage_list, train, use_coeff_codec, &timing, &session, &err);
    if (!prepared) {
        LogError(err);
        return 1;
    }
    bool ok = cfg.eval.linkage_parent_louds_native_eval
                  ? EvaluateRecallLinkageIvfFromDiskTimedParentLOUDSNative(cfg, query_dataset, linkage_list, train, use_coeff_codec, &adc, &timing, &session, &err)
                  : EvaluateRecallLinkageIvfFromDiskTimed(cfg, query_dataset, linkage_list, train, use_coeff_codec, &adc, &timing, &session, &err);
    if (!ok) {
        LogError(err);
        return 1;
    }

    ColMajorMatrix<int> oracle = OracleRerankByRawDistance(cfg, train, query_dataset, adc.indices, &err);
    if (!err.empty()) {
        LogError(err);
        return 1;
    }

    const std::vector<float> adc_curve = RecallCurveTop1(query_dataset.gt, adc.indices);
    const std::vector<float> oracle_curve = RecallCurveTop1(query_dataset.gt, oracle);

    std::ostringstream report;
    report << std::fixed << std::setprecision(6);
    report << "[oracle_rerank] run_root = " << args.run_root << "\n";
    report << "[oracle_rerank] candidate_k = " << args.candidate_k << "\n";
    report << "[oracle_rerank] nq = " << query_dataset.Xq.cols << "\n";
    report << "[oracle_rerank] nprobe = " << cfg.eval.linkage_nprobe << "\n";
    report << "[oracle_rerank] core_wall_sec = " << timing.core_wall_sec << "\n";
    PrintCurveLine(&report, "adc_r", adc_curve, args.candidate_k);
    PrintCurveLine(&report, "oracle_rerank_r", oracle_curve, args.candidate_k);
    report << "[oracle_rerank] oracle_gain_r1 = " << (oracle_curve.empty() || adc_curve.empty() ? 0.0f : oracle_curve[0] - adc_curve[0]) << "\n";
    if (args.candidate_k >= 10 && adc_curve.size() >= 10 && oracle_curve.size() >= 10) {
        report << "[oracle_rerank] oracle_gain_r10 = " << (oracle_curve[9] - adc_curve[9]) << "\n";
    }

    const std::filesystem::path out_path(args.out_report);
    if (out_path.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(out_path.parent_path(), ec);
    }
    std::ofstream out(out_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        LogError("Failed to open report: " + out_path.string());
        return 1;
    }
    const std::string s = report.str();
    out.write(s.data(), static_cast<std::streamsize>(s.size()));
    if (!out) {
        LogError("Failed to write report: " + out_path.string());
        return 1;
    }
    std::cout << s;
    LogInfo("Wrote oracle rerank report: " + out_path.string());
    return 0;
}

