#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <omp.h>

#include "stlq/common/config.h"
#include "stlq/common/logger.h"
#include "stlq/eval/recall_linkage_disk.h"
#include "stlq/io/dataset_reader_factory.h"
#include "stlq/io/linkage_list_store.h"
#include "stlq/io/result_io.h"
#include "stlq/pipeline/app_utils.h"

namespace {

struct Args {
    std::string run_root;
    std::vector<int> candidate_ks{100};
    std::vector<int> nprobes;
    int final_k = 1;
    int rounds = 3;
    int warmup = 1;
    int threads = 32;
};

struct RecallMetrics {
    double r1 = 0.0;
    double r10 = 0.0;
    double r100 = 0.0;
    double top100_set_recall = 0.0;
};

std::string TrimAscii(std::string s) {
    const auto is_ws = [](unsigned char c) {
        return c == ' ' || c == '\t' || c == '\r' || c == '\n';
    };
    while (!s.empty() && is_ws(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
    while (!s.empty() && is_ws(static_cast<unsigned char>(s.back()))) s.pop_back();
    return s;
}

std::string StripMatchingQuotes(std::string s) {
    s = TrimAscii(std::move(s));
    if (s.size() >= 2) {
        const char first = s.front();
        const char last = s.back();
        if ((first == '"' && last == '"') || (first == '\'' && last == '\'')) {
            return s.substr(1, s.size() - 2);
        }
    }
    return s;
}

bool ParsePositiveCsv(const std::string& text, std::vector<int>* out) {
    if (!out) return false;
    out->clear();
    std::istringstream input(text);
    std::string token;
    while (std::getline(input, token, ',')) {
        token = TrimAscii(std::move(token));
        if (token.empty()) continue;
        char* end = nullptr;
        const long value = std::strtol(token.c_str(), &end, 10);
        if (!end || *end != '\0' || value <= 0 || value > std::numeric_limits<int>::max()) {
            return false;
        }
        out->push_back(static_cast<int>(value));
    }
    std::sort(out->begin(), out->end());
    out->erase(std::unique(out->begin(), out->end()), out->end());
    return !out->empty();
}

bool ParseArgs(int argc, char** argv, Args* out, std::string* err) {
    if (!out) return false;
    *out = {};
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        const auto take = [&](std::string* value) -> bool {
            if (i + 1 >= argc) return false;
            *value = argv[++i];
            return true;
        };
        std::string value;
        if (arg == "--run_root" || arg == "--run-root") {
            if (!take(&out->run_root)) {
                if (err) *err = "Missing value for " + arg;
                return false;
            }
        } else if (arg == "--candidate_ks" || arg == "--candidate-ks") {
            if (!take(&value) || !ParsePositiveCsv(value, &out->candidate_ks)) {
                if (err) *err = "Invalid value for " + arg;
                return false;
            }
        } else if (arg == "--nprobes") {
            if (!take(&value) || !ParsePositiveCsv(value, &out->nprobes)) {
                if (err) *err = "Invalid value for " + arg;
                return false;
            }
        } else if (arg == "--final_k" || arg == "--final-k") {
            if (!take(&value)) {
                if (err) *err = "Missing value for " + arg;
                return false;
            }
            out->final_k = std::atoi(value.c_str());
        } else if (arg == "--rounds") {
            if (!take(&value)) {
                if (err) *err = "Missing value for " + arg;
                return false;
            }
            out->rounds = std::atoi(value.c_str());
        } else if (arg == "--warmup") {
            if (!take(&value)) {
                if (err) *err = "Missing value for " + arg;
                return false;
            }
            out->warmup = std::atoi(value.c_str());
        } else if (arg == "--threads") {
            if (!take(&value)) {
                if (err) *err = "Missing value for " + arg;
                return false;
            }
            out->threads = std::atoi(value.c_str());
        } else if (arg == "-h" || arg == "--help") {
            std::cout
                << "bench_stlq_exact_rerank\n\n"
                << "Required:\n"
                << "  --run_root <dir>          STLQ run containing config_snapshot and linkage_list\n\n"
                << "Optional:\n"
                << "  --candidate_ks <csv>      ADC shortlist sizes (default: 100)\n"
                << "  --nprobes <csv>           IVF probes (default: value from snapshot)\n"
                << "  --final_k <int>           Exact output size (default: 1)\n"
                << "  --rounds <int>            Measured rounds (default: 3)\n"
                << "  --warmup <int>            Warmup rounds (default: 1)\n"
                << "  --threads <int>           OpenMP threads (default: 32)\n";
            std::exit(0);
        } else {
            if (err) *err = "Unknown argument: " + arg;
            return false;
        }
    }
    if (out->run_root.empty()) {
        if (err) *err = "--run_root is required";
        return false;
    }
    if (out->final_k <= 0 || out->rounds <= 0 || out->warmup < 0 || out->threads <= 0) {
        if (err) *err = "final_k, rounds, and threads must be positive; warmup must be nonnegative";
        return false;
    }
    for (int candidate_k : out->candidate_ks) {
        if (candidate_k < out->final_k) {
            if (err) *err = "Every candidate_k must be >= final_k";
            return false;
        }
    }
    return true;
}

std::filesystem::path PickConfigSnapshotPath(const std::filesystem::path& run_root) {
    for (const char* name : {"config_snapshot.txt", "config_snapshot.cfg", "basic_linux.cfg"}) {
        const std::filesystem::path path = run_root / name;
        if (std::filesystem::exists(path)) return path;
    }
    return run_root / "config_snapshot.txt";
}

std::string ReadRunStateValue(const std::filesystem::path& run_root, const std::string& key) {
    std::ifstream input(run_root / "run_state.txt");
    if (!input.is_open()) return {};
    const std::string prefix = key + " = ";
    std::string line;
    while (std::getline(input, line)) {
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
    std::string err;
    return !stlq::io::MakeDatedPathForLoad(base_path, load_date, load_seq, &err).empty();
}

std::string ResolveLoadBasePath(const std::string& base_path,
                                const std::string& load_date,
                                const std::string& load_seq,
                                const std::filesystem::path& run_root,
                                const std::filesystem::path& cfg_path) {
    if (base_path.empty() || PathExistsForLoadMode(base_path, load_date, load_seq)) return base_path;
    const std::filesystem::path input(base_path);
    if (input.is_absolute()) return base_path;
    const std::string stripped =
        base_path.rfind("./", 0) == 0 ? base_path.substr(2) : base_path;
    std::vector<std::filesystem::path> roots{
        std::filesystem::current_path(), run_root, cfg_path.parent_path()};
    const auto append_ancestors = [&](std::filesystem::path current) {
        while (!current.empty()) {
            roots.push_back(current);
            const std::filesystem::path parent = current.parent_path();
            if (parent == current) break;
            current = parent;
        }
    };
    append_ancestors(run_root.parent_path());
    append_ancestors(cfg_path.parent_path());
    for (const auto& root : roots) {
        if (root.empty()) continue;
        const std::string candidate = (root / stripped).string();
        if (PathExistsForLoadMode(candidate, load_date, load_seq)) return candidate;
    }
    return base_path;
}

double Median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t mid = values.size() / 2;
    return (values.size() & 1U) != 0U
               ? values[mid]
               : 0.5 * (values[mid - 1] + values[mid]);
}

float SquaredL2(const float* query, const float* base, int d) {
    float sum = 0.0f;
#pragma omp simd reduction(+ : sum)
    for (int j = 0; j < d; ++j) {
        const float diff = query[j] - base[j];
        sum += diff * diff;
    }
    return sum;
}

double ExactRerank(const stlq::ColMajorMatrix<float>& raw_queries,
                   const stlq::ColMajorMatrix<float>& raw_base,
                   const stlq::ColMajorMatrix<int>& candidates,
                   int final_k,
                   int threads,
                   stlq::ColMajorMatrix<int>* out) {
    const int nq = raw_queries.cols;
    const int d = raw_queries.rows;
    const int candidate_k = candidates.rows;
    *out = stlq::ColMajorMatrix<int>(final_k, nq);
    const double start = omp_get_wtime();
#pragma omp parallel num_threads(threads)
    {
        std::vector<std::pair<float, int>> scored;
        if (final_k > 1) scored.reserve(static_cast<std::size_t>(candidate_k));
#pragma omp for schedule(static)
        for (int q = 0; q < nq; ++q) {
            const float* query = raw_queries.Col(q);
            int* result = out->Col(q);
            if (final_k == 1) {
                float best_dist = std::numeric_limits<float>::infinity();
                int best_id = -1;
                for (int rank = 0; rank < candidate_k; ++rank) {
                    const int id = candidates(rank, q);
                    if (id < 0 || id >= raw_base.cols) continue;
                    const float dist = SquaredL2(query, raw_base.Col(id), d);
                    if (dist < best_dist || (dist == best_dist && id < best_id)) {
                        best_dist = dist;
                        best_id = id;
                    }
                }
                result[0] = best_id;
                continue;
            }

            scored.clear();
            for (int rank = 0; rank < candidate_k; ++rank) {
                const int id = candidates(rank, q);
                if (id < 0 || id >= raw_base.cols) continue;
                scored.emplace_back(SquaredL2(query, raw_base.Col(id), d), id);
            }
            const auto less = [](const auto& lhs, const auto& rhs) {
                return lhs.first < rhs.first ||
                       (lhs.first == rhs.first && lhs.second < rhs.second);
            };
            if (static_cast<int>(scored.size()) > final_k) {
                std::nth_element(scored.begin(), scored.begin() + final_k, scored.end(), less);
                scored.resize(static_cast<std::size_t>(final_k));
            }
            std::sort(scored.begin(), scored.end(), less);
            int rank = 0;
            for (; rank < static_cast<int>(scored.size()); ++rank) {
                result[rank] = scored[static_cast<std::size_t>(rank)].second;
            }
            for (; rank < final_k; ++rank) result[rank] = -1;
        }
    }
    return omp_get_wtime() - start;
}

RecallMetrics ComputeRecall(const std::vector<int>& gt_first,
                            const stlq::ColMajorMatrix<int>* gt_top100,
                            const stlq::ColMajorMatrix<int>& results) {
    RecallMetrics metrics;
    const int nq = results.cols;
    const int result_k = results.rows;
    std::uint64_t hit1 = 0;
    std::uint64_t hit10 = 0;
    std::uint64_t hit100 = 0;
    std::uint64_t set100_hits = 0;
#pragma omp parallel for schedule(static) reduction(+ : hit1, hit10, hit100, set100_hits)
    for (int q = 0; q < nq; ++q) {
        const int gt = gt_first[static_cast<std::size_t>(q)];
        const int* ids = results.Col(q);
        for (int rank = 0; rank < result_k; ++rank) {
            if (ids[rank] == gt) {
                hit1 += rank < 1;
                hit10 += rank < 10;
                hit100 += rank < 100;
                break;
            }
        }
        if (result_k >= 100 && gt_top100 && gt_top100->rows >= 100) {
            const int* gt_ids = gt_top100->Col(q);
            for (int rank = 0; rank < 100; ++rank) {
                const int id = ids[rank];
                for (int gt_rank = 0; gt_rank < 100; ++gt_rank) {
                    if (id == gt_ids[gt_rank]) {
                        ++set100_hits;
                        break;
                    }
                }
            }
        }
    }
    const double denom = static_cast<double>(nq);
    metrics.r1 = static_cast<double>(hit1) / denom;
    metrics.r10 = static_cast<double>(hit10) / denom;
    metrics.r100 = static_cast<double>(hit100) / denom;
    if (result_k >= 100 && gt_top100 && gt_top100->rows >= 100) {
        metrics.top100_set_recall =
            static_cast<double>(set100_hits) / (denom * 100.0);
    }
    return metrics;
}

double ComputeGtFirstCoverage(const std::vector<int>& gt_first,
                              const stlq::ColMajorMatrix<int>& candidates) {
    std::uint64_t hits = 0;
#pragma omp parallel for schedule(static) reduction(+ : hits)
    for (int q = 0; q < candidates.cols; ++q) {
        const int gt = gt_first[static_cast<std::size_t>(q)];
        const int* ids = candidates.Col(q);
        for (int rank = 0; rank < candidates.rows; ++rank) {
            if (ids[rank] == gt) {
                ++hits;
                break;
            }
        }
    }
    return static_cast<double>(hits) / static_cast<double>(candidates.cols);
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
    std::vector<int> ignored_nprobes;
    if (!LoadConfigFile(cfg_path.string(), &cfg, &err, &ignored_nprobes)) {
        LogError(err);
        return 1;
    }
    if (args.nprobes.empty()) args.nprobes = {std::max(1, cfg.eval.linkage_nprobe)};
    cfg.eval.linkage_enabled = true;
    cfg.eval.linkage_use_ivf_disk = true;
    cfg.eval.linkage_coeff_mode = "int8";
    cfg.eval.bench_quiet = true;
    cfg.runtime.omp_threads = args.threads;
    omp_set_num_threads(args.threads);

    TrainResult train;
    {
        std::string base_path = cfg.io.train_file;
        std::string load_date = cfg.io.load_date;
        std::string load_seq = cfg.io.load_seq;
        const std::string effective_train_h5 =
            ReadRunStateValue(run_root, "effective_train_h5");
        if (!effective_train_h5.empty() && effective_train_h5 != "<none>") {
            base_path = effective_train_h5;
            load_date = "raw";
            load_seq.clear();
        }
        base_path = ResolveLoadBasePath(base_path, load_date, load_seq, run_root, cfg_path);
        if (!io::LoadTrainResults(
                base_path, load_date, load_seq, cfg.model.m, &train, &err)) {
            LogError(err.empty() ? "LoadTrainResults failed." : err);
            return 1;
        }
    }

    Dataset query_dataset;
    if (!io::LoadQuerySet(cfg, &query_dataset.Xq, &query_dataset.gt, &err)) {
        LogError(err);
        return 1;
    }
    const ColMajorMatrix<float> raw_queries = query_dataset.Xq;
    if (!IsIdentityRotation(train.R)) ApplyRotationInPlace(train.R, &query_dataset.Xq);

    ColMajorMatrix<int> gt_top100;
    if (args.final_k >= 100 &&
        !io::LoadGroundtruthTopK(cfg, query_dataset.Xq.cols, 100, &gt_top100, &err)) {
        LogError(err);
        return 1;
    }

    io::DatasetVectorReader base_reader;
    if (!io::OpenDatasetVectorReader(cfg, io::DatasetRole::kBase, &base_reader, &err)) {
        LogError(err);
        return 1;
    }
    const std::uint64_t requested_nbase =
        std::min<std::uint64_t>(base_reader.n, static_cast<std::uint64_t>(cfg.dataset.nbase));
    if (requested_nbase == 0 ||
        requested_nbase > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
        LogError("The in-memory exact-rerank benchmark requires 1..UINT32_MAX base vectors.");
        return 1;
    }
    ColMajorMatrix<float> raw_base;
    if (!io::ReadDatasetVectorBlockF32(
            base_reader, 0, static_cast<std::uint32_t>(requested_nbase), &raw_base, &err)) {
        LogError(err);
        return 1;
    }
    if (raw_base.rows != raw_queries.rows) {
        LogError("Base/query dimension mismatch.");
        return 1;
    }

    io::LinkageListReader linkage_list;
    if (!linkage_list.Open((run_root / "linkage_list").string(), &err)) {
        LogError(err);
        return 1;
    }

    DiskLinkageEvalSession session{};
    DiskLinkageEvalTiming prepare_timing{};
    const bool use_coeff_codec = true;
    const bool prepared = cfg.eval.linkage_parent_louds_native_eval
                              ? PrepareRecallLinkageIvfDiskSessionParentLOUDSNative(
                                    cfg, linkage_list, train, use_coeff_codec,
                                    &prepare_timing, &session, &err)
                              : PrepareRecallLinkageIvfDiskSession(
                                    cfg, linkage_list, train, use_coeff_codec,
                                    &prepare_timing, &session, &err);
    if (!prepared) {
        LogError(err);
        return 1;
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "queries=" << query_dataset.Xq.cols
              << " dim=" << query_dataset.Xq.rows
              << " base=" << raw_base.cols
              << " threads=" << args.threads
              << " final_k=" << args.final_k
              << " raw_base=f32_in_memory\n";
    std::cout << "QPS includes STLQ routing/query tables/scan/top-k and exact raw-vector L2 rerank\n";
    std::cout
        << "candidate_k\tnprobe\tmedian_qps_e2e\tavg_qps_e2e"
           "\tmedian_scan_sec\tmedian_rerank_sec\tmedian_e2e_sec"
           "\tcandidate_gt1_recall\tR@1\tR@10\tR@100"
           "\tTop100SetRecall\traw_MiB_per_query\n";

    for (int candidate_k : args.candidate_ks) {
        cfg.dataset.k = candidate_k;
        cfg.dataset.k_set = true;
        for (int nprobe : args.nprobes) {
            cfg.eval.linkage_nprobe = nprobe;
            RecallResult adc;
            ColMajorMatrix<int> reranked;

            const auto run_once = [&](double* scan_sec, double* rerank_sec) -> bool {
                DiskLinkageEvalTiming timing{};
                const bool ok = cfg.eval.linkage_parent_louds_native_eval
                                    ? EvaluateRecallLinkageIvfFromDiskTimedParentLOUDSNative(
                                          cfg, query_dataset, linkage_list, train,
                                          use_coeff_codec, &adc, &timing, &session, &err)
                                    : EvaluateRecallLinkageIvfFromDiskTimed(
                                          cfg, query_dataset, linkage_list, train,
                                          use_coeff_codec, &adc, &timing, &session, &err);
                if (!ok) return false;
                *scan_sec = timing.core_wall_sec;
                *rerank_sec = ExactRerank(
                    raw_queries, raw_base, adc.indices, args.final_k, args.threads, &reranked);
                return true;
            };

            for (int round = 0; round < args.warmup; ++round) {
                double scan_sec = 0.0;
                double rerank_sec = 0.0;
                if (!run_once(&scan_sec, &rerank_sec)) {
                    LogError(err);
                    return 1;
                }
            }

            std::vector<double> scan_times;
            std::vector<double> rerank_times;
            std::vector<double> e2e_times;
            std::vector<double> qps_values;
            scan_times.reserve(static_cast<std::size_t>(args.rounds));
            rerank_times.reserve(static_cast<std::size_t>(args.rounds));
            e2e_times.reserve(static_cast<std::size_t>(args.rounds));
            qps_values.reserve(static_cast<std::size_t>(args.rounds));
            for (int round = 0; round < args.rounds; ++round) {
                double scan_sec = 0.0;
                double rerank_sec = 0.0;
                if (!run_once(&scan_sec, &rerank_sec)) {
                    LogError(err);
                    return 1;
                }
                const double e2e_sec = scan_sec + rerank_sec;
                scan_times.push_back(scan_sec);
                rerank_times.push_back(rerank_sec);
                e2e_times.push_back(e2e_sec);
                qps_values.push_back(
                    static_cast<double>(query_dataset.Xq.cols) / e2e_sec);
            }

            const double candidate_gt1_recall =
                ComputeGtFirstCoverage(query_dataset.gt, adc.indices);
            const RecallMetrics metrics =
                ComputeRecall(query_dataset.gt,
                              args.final_k >= 100 ? &gt_top100 : nullptr,
                              reranked);
            const double avg_qps =
                std::accumulate(qps_values.begin(), qps_values.end(), 0.0) /
                static_cast<double>(qps_values.size());
            const double median_e2e_sec = Median(e2e_times);
            const double raw_mib =
                static_cast<double>(candidate_k) * raw_base.rows * sizeof(float) /
                (1024.0 * 1024.0);

            std::cout << candidate_k << '\t'
                      << nprobe << '\t'
                      << static_cast<double>(query_dataset.Xq.cols) / median_e2e_sec << '\t'
                      << avg_qps << '\t'
                      << Median(scan_times) << '\t'
                      << Median(rerank_times) << '\t'
                      << median_e2e_sec << '\t'
                      << candidate_gt1_recall << '\t'
                      << metrics.r1 << '\t'
                      << metrics.r10 << '\t'
                      << metrics.r100 << '\t'
                      << metrics.top100_set_recall << '\t'
                      << raw_mib << '\n';
        }
    }
    return 0;
}
