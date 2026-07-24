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
#include <unordered_map>
#include <utility>
#include <vector>

#include "stlq/common/config.h"
#include "stlq/common/logger.h"
#include "stlq/eval/linkage_cluster_provider.h"
#include "stlq/eval/recall_linkage_disk.h"
#include "stlq/io/dataset_reader_factory.h"
#include "stlq/io/linkage_coeff_codec_store.h"
#include "stlq/io/linkage_list_store.h"
#include "stlq/io/result_io.h"
#include "stlq/pipeline/app_utils.h"
#include "stlq/quantizer/codebook_meta.h"

namespace {

struct Args {
    std::string run_root;
    std::string out_csv;
    int candidate_k = 100;
};

struct NodeFeature {
    int cid = -1;
    int local_id = -1;
    int real_pos = -1;
    int depth = -1;
    int parent_local_id = -1;
    int parent_depth = -1;
    int parent_kind = 0;  // 0 root, 1 virtual, 2 real_prev, 3 real_same, 4 real_skip, 5 invalid
    int child_degree = 0;
    int subtree_real = 0;
    int cluster_n_real = 0;
    int cluster_n_virtual = 0;
    int cluster_max_depth = 0;
    int cluster_max_child_degree = 0;
    int cluster_max_subtree_real = 0;
    double cluster_mean_depth = 0.0;
    double cluster_degree_gini = 0.0;
    double cluster_linked_ratio = 0.0;
};

struct ReconContext {
    stlq::io::LinkageCoeffCodecReader coeff_reader;
    stlq::eval::ClusterProvider provider;
    const stlq::ColMajorMatrix<float>* C_root0 = nullptr;
    stlq::CodebookMeta meta_root_small;
    stlq::CodebookMeta meta_one;
    std::vector<stlq::eval::ClusterView> cv_by_cid;
    std::vector<unsigned char> cv_loaded;
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
        const std::string a(argv[i] ? argv[i] : "");
        auto take = [&](std::string* dst) -> bool {
            if (i + 1 >= argc || !argv[i + 1]) return false;
            *dst = argv[++i];
            return true;
        };
        if (a == "--run_root" || a == "--run-root") {
            if (!take(&out->run_root)) {
                if (err) *err = "Missing value for " + a;
                return false;
            }
        } else if (a == "--out" || a == "--out_csv" || a == "--out-csv") {
            if (!take(&out->out_csv)) {
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
                << "export_stlq_candidate_features\n\n"
                << "Required:\n"
                << "  --run_root <dir>       STLQ run root containing config_snapshot.txt and linkage_list/\n\n"
                << "Optional:\n"
                << "  --candidate_k <int>    STLQ ADC shortlist size to export (default: 100)\n"
                << "  --out <csv>            Output CSV (default: <run_root>/analysis/candidate_features_k<K>.csv)\n";
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
    if (out->out_csv.empty()) {
        out->out_csv = (std::filesystem::path(out->run_root) / "analysis" /
                        ("candidate_features_k" + std::to_string(out->candidate_k) + ".csv"))
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

double Dot(const float* a, const float* b, int d) {
    double s = 0.0;
    for (int i = 0; i < d; ++i) {
        s += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return s;
}

double Norm2(const float* x, int d) {
    return Dot(x, x, d);
}

template <typename T>
bool ReadBinaryFileExact(const std::filesystem::path& path,
                         std::size_t count,
                         std::vector<T>* out,
                         std::string* err) {
    if (!out) return false;
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        if (err) *err = "failed to open: " + path.string();
        return false;
    }
    out->assign(count, T{});
    if (count == 0) return true;
    in.read(reinterpret_cast<char*>(out->data()), static_cast<std::streamsize>(count * sizeof(T)));
    if (!in) {
        if (err) *err = "failed to read: " + path.string();
        return false;
    }
    return true;
}

void AddScaledVectorF32(float alpha, const float* src, int d, float* dst) {
    for (int r = 0; r < d; ++r) dst[r] += alpha * src[r];
}

std::uint32_t ReadParent1BasedAt(const stlq::eval::ClusterView& cv, int pos) {
    if (pos < 0 || pos >= cv.n_real) return 0u;
    if (cv.parent_is_u16) return cv.parent_1based_u16 ? static_cast<std::uint32_t>(cv.parent_1based_u16[pos]) : 0u;
    return cv.parent_1based ? cv.parent_1based[pos] : 0u;
}

double Gini(std::vector<double> xs) {
    if (xs.empty()) return 0.0;
    std::sort(xs.begin(), xs.end());
    const double sum = std::accumulate(xs.begin(), xs.end(), 0.0);
    if (sum <= 0.0) return 0.0;
    long double weighted = 0.0;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        weighted += static_cast<long double>(i + 1) * static_cast<long double>(xs[i]);
    }
    const long double n = static_cast<long double>(xs.size());
    return static_cast<double>((2.0L * weighted) / (n * static_cast<long double>(sum)) - (n + 1.0L) / n);
}

bool ReconstructRealNodeCoeffCodec(const stlq::eval::ClusterView& cv,
                                   int cid,
                                   int local_pos,
                                   const stlq::ColMajorMatrix<float>& C_root0,
                                   const stlq::CodebookMeta& meta_root_small,
                                   const stlq::CodebookMeta& meta_one,
                                   std::vector<float>* out_xhat) {
    if (!out_xhat) return false;
    out_xhat->clear();
    if (local_pos < 0 || local_pos >= cv.n_real) return false;
    if (!cv.q_layer_major || !cv.scales_root || !cv.scales_linkage) return false;
    if (!cv.codes_small_bytes || !cv.code0_one_bytes) return false;
    if (cv.n_virt > 0 && !cv.virt_codes_small_bytes) return false;
    if (cv.m <= 0 || cv.m_codes != std::max(0, cv.m - 1)) return false;
    if (cv.nc != cv.n_real + cv.n_virt) return false;
    if (cid < 0 || cid >= C_root0.cols || C_root0.rows <= 0) return false;
    if (meta_one.flat.rows != C_root0.rows || meta_root_small.flat.rows != C_root0.rows) return false;

    const int d = C_root0.rows;
    const int m = cv.m;
    const int stride_codes = cv.m_codes;
    const int n_virt = cv.n_virt;
    const int nc = cv.nc;

    auto q_i8 = [&](int layer, int local_idx) -> float {
        return static_cast<float>(cv.q_layer_major[static_cast<std::size_t>(layer) * static_cast<std::size_t>(nc) +
                                                   static_cast<std::size_t>(local_idx)]);
    };

    int local = n_virt + local_pos;
    if (local < 0 || local >= nc) return false;
    std::vector<int> linkage_stack;
    linkage_stack.reserve(32);
    while (true) {
        if (local < n_virt) break;
        const int pos = local - n_virt;
        if (pos < 0 || pos >= cv.n_real) return false;
        if (pos < cv.n_root_real) break;
        linkage_stack.push_back(local);
        const std::uint32_t p1 = ReadParent1BasedAt(cv, pos);
        if (p1 == 0u) break;
        local = static_cast<int>(p1) - 1;
        if (local < 0 || local >= nc) return false;
    }
    const int root_local = local;

    out_xhat->assign(static_cast<std::size_t>(d), 0.0f);
    float* xhat = out_xhat->data();

    {
        const float a0 = cv.scales_root[0] * q_i8(0, root_local);
        AddScaledVectorF32(a0, C_root0.Col(cid), d, xhat);
        for (int l = 1; l < m; ++l) {
            const int off = meta_root_small.offsets[static_cast<std::size_t>(l - 1)];
            const std::uint8_t code = (root_local < n_virt)
                ? cv.virt_codes_small_bytes[static_cast<std::size_t>(root_local) * static_cast<std::size_t>(stride_codes) +
                                            static_cast<std::size_t>(l - 1)]
                : cv.codes_small_bytes[static_cast<std::size_t>(root_local - n_virt) * static_cast<std::size_t>(stride_codes) +
                                       static_cast<std::size_t>(l - 1)];
            const float a = cv.scales_root[l] * q_i8(l, root_local);
            AddScaledVectorF32(a, meta_root_small.flat.Col(off + static_cast<int>(code)), d, xhat);
        }
    }

    while (!linkage_stack.empty()) {
        const int child_local = linkage_stack.back();
        linkage_stack.pop_back();
        const int pos = child_local - n_virt;
        if (pos < 0 || pos >= cv.n_real) return false;

        const std::uint8_t code0 = cv.code0_one_bytes[pos];
        const float a0 = cv.scales_linkage[0] * q_i8(0, child_local);
        AddScaledVectorF32(a0, meta_one.flat.Col(meta_one.offsets[0] + static_cast<int>(code0)), d, xhat);

        const std::uint8_t* row =
            cv.codes_small_bytes + static_cast<std::size_t>(pos) * static_cast<std::size_t>(stride_codes);
        for (int l = 1; l < m; ++l) {
            const std::uint8_t code = row[l - 1];
            const float a = cv.scales_linkage[l] * q_i8(l, child_local);
            AddScaledVectorF32(a, meta_one.flat.Col(meta_one.offsets[static_cast<std::size_t>(l)] + static_cast<int>(code)), d, xhat);
        }
    }
    return true;
}

std::vector<int> BuildRealDepth(const std::vector<std::uint32_t>& depth_offsets, int n_real) {
    std::vector<int> real_depth(static_cast<std::size_t>(n_real), 0);
    for (std::size_t dep = 0; dep + 1 < depth_offsets.size(); ++dep) {
        const int lo = static_cast<int>(depth_offsets[dep]);
        const int hi = static_cast<int>(depth_offsets[dep + 1]);
        for (int i = std::max(0, lo); i < std::min(n_real, hi); ++i) {
            real_depth[static_cast<std::size_t>(i)] = static_cast<int>(dep);
        }
    }
    return real_depth;
}

int ParentKind(int child_depth,
               int parent_local,
               int n_virt,
               int n_real,
               const std::vector<int>& real_depth,
               int* parent_depth) {
    if (parent_depth) *parent_depth = -1;
    if (parent_local < 0) return 0;
    if (parent_local < n_virt) return 1;
    if (parent_local >= n_virt + n_real) return 5;
    const int p_real = parent_local - n_virt;
    const int p_depth = real_depth[static_cast<std::size_t>(p_real)];
    if (parent_depth) *parent_depth = p_depth;
    const int delta = child_depth - p_depth;
    if (delta == 1) return 2;
    if (delta == 0) return 3;
    if (delta > 1) return 4;
    return 5;
}

const char* ParentKindName(int kind) {
    switch (kind) {
    case 0: return "super_root";
    case 1: return "virtual";
    case 2: return "real_prev_depth";
    case 3: return "real_same_depth";
    case 4: return "real_skip_depth";
    default: return "real_deeper_or_invalid";
    }
}

bool BuildNodeFeatureMap(const stlq::io::LinkageListReader& linkage_list,
                         std::unordered_map<std::uint32_t, NodeFeature>* out,
                         std::string* err) {
    if (!out) return false;
    out->clear();
    out->reserve(static_cast<std::size_t>(linkage_list.total_real()));

    for (int cid = 0; cid < linkage_list.nlist(); ++cid) {
        std::uint64_t real_lo = 0, real_hi = 0, virt_lo = 0, virt_hi = 0, depth_lo = 0, depth_hi = 0;
        if (!linkage_list.ReadClusterSpan(cid, &real_lo, &real_hi, &virt_lo, &virt_hi, &depth_lo, &depth_hi, err)) {
            return false;
        }
        const int n_real = static_cast<int>(real_hi - real_lo);
        const int n_virt = static_cast<int>(virt_hi - virt_lo);
        if (n_real <= 0) continue;

        std::vector<std::uint32_t> real_ids;
        std::vector<std::uint32_t> parent;
        std::vector<std::uint32_t> depth_offsets;
        std::vector<std::uint8_t> codes;
        std::vector<float> coeffs;
        std::vector<std::uint8_t> code0;
        std::vector<float> a0;
        std::vector<std::uint8_t> virt_codes;
        std::vector<float> virt_coeffs;
        std::vector<float> virt_a0;
        if (!linkage_list.ReadCluster(cid, &real_ids, &parent, &depth_offsets, &codes, &coeffs,
                                      &code0, &a0, &virt_codes, &virt_coeffs, &virt_a0, err)) {
            return false;
        }
        if (static_cast<int>(real_ids.size()) != n_real ||
            static_cast<int>(parent.size()) != n_real ||
            depth_offsets.size() < 2 ||
            depth_offsets.back() != static_cast<std::uint32_t>(n_real)) {
            if (err) *err = "BuildNodeFeatureMap: invalid cluster payload at cid=" + std::to_string(cid);
            return false;
        }

        const std::vector<int> real_depth = BuildRealDepth(depth_offsets, n_real);
        std::vector<std::uint32_t> degree(static_cast<std::size_t>(n_real + n_virt), 0);
        std::vector<std::uint32_t> subtree(static_cast<std::size_t>(n_real + n_virt), 0);
        int max_depth = 0;
        std::uint64_t depth_sum = 0;
        const std::uint32_t n_root = depth_offsets[1] - depth_offsets[0];

        for (int i = 0; i < n_real; ++i) {
            const int local = n_virt + i;
            subtree[static_cast<std::size_t>(local)] = 1;
            const int dep = real_depth[static_cast<std::size_t>(i)];
            max_depth = std::max(max_depth, dep);
            depth_sum += static_cast<std::uint64_t>(dep);
            const std::uint32_t p1 = parent[static_cast<std::size_t>(i)];
            if (p1 == 0) continue;
            const int p_local = static_cast<int>(p1) - 1;
            if (p_local >= 0 && p_local < n_real + n_virt) {
                degree[static_cast<std::size_t>(p_local)] += 1;
            }
        }
        for (int i = n_real - 1; i >= 0; --i) {
            const std::uint32_t p1 = parent[static_cast<std::size_t>(i)];
            if (p1 == 0) continue;
            const int child_local = n_virt + i;
            const int p_local = static_cast<int>(p1) - 1;
            if (p_local >= 0 && p_local < n_real + n_virt) {
                subtree[static_cast<std::size_t>(p_local)] += subtree[static_cast<std::size_t>(child_local)];
            }
        }

        std::uint32_t max_degree = 0;
        std::uint32_t max_subtree = 0;
        std::vector<double> cluster_degrees;
        cluster_degrees.reserve(static_cast<std::size_t>(n_real + n_virt));
        for (int local = 0; local < n_real + n_virt; ++local) {
            const std::uint32_t deg = degree[static_cast<std::size_t>(local)];
            const std::uint32_t sub = subtree[static_cast<std::size_t>(local)];
            max_degree = std::max(max_degree, deg);
            max_subtree = std::max(max_subtree, sub);
            cluster_degrees.push_back(static_cast<double>(deg));
        }
        const double mean_depth = static_cast<double>(depth_sum) / static_cast<double>(n_real);
        const double degree_gini = Gini(cluster_degrees);
        const double linked_ratio = 1.0 - static_cast<double>(n_root) / static_cast<double>(n_real);

        for (int i = 0; i < n_real; ++i) {
            const int local = n_virt + i;
            const int dep = real_depth[static_cast<std::size_t>(i)];
            const std::uint32_t p1 = parent[static_cast<std::size_t>(i)];
            const int parent_local = (p1 == 0) ? -1 : static_cast<int>(p1) - 1;
            int parent_depth = -1;
            const int parent_kind = ParentKind(dep, parent_local, n_virt, n_real, real_depth, &parent_depth);

            NodeFeature f;
            f.cid = cid;
            f.local_id = local;
            f.real_pos = i;
            f.depth = dep;
            f.parent_local_id = parent_local;
            f.parent_depth = parent_depth;
            f.parent_kind = parent_kind;
            f.child_degree = static_cast<int>(degree[static_cast<std::size_t>(local)]);
            f.subtree_real = static_cast<int>(subtree[static_cast<std::size_t>(local)]);
            f.cluster_n_real = n_real;
            f.cluster_n_virtual = n_virt;
            f.cluster_max_depth = max_depth;
            f.cluster_max_child_degree = static_cast<int>(max_degree);
            f.cluster_max_subtree_real = static_cast<int>(max_subtree);
            f.cluster_mean_depth = mean_depth;
            f.cluster_degree_gini = degree_gini;
            f.cluster_linked_ratio = linked_ratio;
            (*out)[real_ids[static_cast<std::size_t>(i)]] = f;
        }
    }
    return true;
}

bool LoadTrainResultForRun(const stlq::Config& cfg,
                           const std::filesystem::path& run_root,
                           const std::filesystem::path& cfg_path,
                           stlq::TrainResult* out,
                           std::string* err) {
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
    return stlq::io::LoadTrainResults(base_path, load_date, load_seq, cfg.model.m, out, err);
}

bool RunLinkageEval(const Args& args,
                    const std::filesystem::path& run_root,
                    const std::filesystem::path& cfg_path,
                    stlq::Config* cfg,
                    stlq::TrainResult* train,
                    stlq::Dataset* query_dataset,
                    stlq::io::LinkageListReader* linkage_list,
                    stlq::RecallResult* adc,
                    std::string* err) {
    using namespace stlq;
    *cfg = DefaultConfig(true);
    NormalizeHVec(cfg);
    std::vector<int> ignored;
    if (!LoadConfigFile(cfg_path.string(), cfg, err, &ignored)) return false;
    cfg->dataset.k = std::max(1, args.candidate_k);
    cfg->dataset.k_set = true;
    cfg->eval.linkage_enabled = true;
    cfg->eval.linkage_use_ivf_disk = true;
    cfg->eval.linkage_repeat = 1;
    cfg->eval.linkage_warmup = 0;
    cfg->eval.linkage_coeff_mode = "int8";
    cfg->eval.bench_quiet = true;

    if (!LoadTrainResultForRun(*cfg, run_root, cfg_path, train, err)) return false;
    if (!io::LoadQuerySet(*cfg, &query_dataset->Xq, &query_dataset->gt, err)) return false;
    if (!IsIdentityRotation(train->R)) ApplyRotationInPlace(train->R, &query_dataset->Xq);

    if (!linkage_list->Open((run_root / "linkage_list").string(), err)) return false;

    DiskLinkageEvalTiming timing{};
    DiskLinkageEvalSession session{};
    const bool use_coeff_codec = true;
    const bool prepared = cfg->eval.linkage_parent_louds_native_eval
        ? PrepareRecallLinkageIvfDiskSessionParentLOUDSNative(*cfg, *linkage_list, *train, use_coeff_codec, &timing, &session, err)
        : PrepareRecallLinkageIvfDiskSession(*cfg, *linkage_list, *train, use_coeff_codec, &timing, &session, err);
    if (!prepared) return false;
    return cfg->eval.linkage_parent_louds_native_eval
        ? EvaluateRecallLinkageIvfFromDiskTimedParentLOUDSNative(*cfg, *query_dataset, *linkage_list, *train, use_coeff_codec, adc, &timing, &session, err)
        : EvaluateRecallLinkageIvfFromDiskTimed(*cfg, *query_dataset, *linkage_list, *train, use_coeff_codec, adc, &timing, &session, err);
}

bool OpenReconContext(const std::filesystem::path& run_root,
                      const stlq::Config& cfg,
                      const stlq::TrainResult& train,
                      const stlq::io::LinkageListReader& linkage_list,
                      ReconContext* out,
                      std::string* err) {
    if (!out) return false;
    if (train.C_root.books.empty() || train.C_one.books.empty()) {
        if (err) *err = "TrainResult missing C_root/C_one.";
        return false;
    }
    const std::filesystem::path linkage_list_dir = run_root / "linkage_list";
    const std::filesystem::path coeff_meta = linkage_list_dir / "coeff_meta.bin";
    if (!std::filesystem::exists(coeff_meta)) {
        if (err) *err = "coeff_meta.bin not found; STLQ reconstruction decomposition requires coeff codec artifacts.";
        return false;
    }
    if (!out->coeff_reader.Open(linkage_list_dir.string(), err)) return false;

    const std::filesystem::path norm2_direct_f32 = linkage_list_dir / "norm2_int8.f32";
    const std::filesystem::path norm2_lut_codes_u8 = linkage_list_dir / "norm2_int8_lut_codes.u8";
    const std::filesystem::path norm2_lut_centers_f32 = linkage_list_dir / "norm2_int8_lut_centers.f32";
    const bool has_norm2_direct_f32 = std::filesystem::exists(norm2_direct_f32);
    const bool has_norm2_lut = std::filesystem::exists(norm2_lut_codes_u8) &&
                               std::filesystem::exists(norm2_lut_centers_f32);
    const bool use_norm2_lut = (!has_norm2_direct_f32) && has_norm2_lut;

    std::uint32_t norm2_lut_h = 256;
    if (use_norm2_lut) {
        std::error_code ec;
        const std::uintmax_t bytes = std::filesystem::file_size(norm2_lut_centers_f32, ec);
        if (!ec && bytes > 0 && (bytes % sizeof(float) == 0)) {
            norm2_lut_h = static_cast<std::uint32_t>(bytes / sizeof(float));
        }
    }

    if (!out->provider.Open(linkage_list,
                            &out->coeff_reader,
                            nullptr,
                            true,
                            true,
                            static_cast<std::uint32_t>(cfg.eval.parent_louds_select_stride),
                            static_cast<std::uint32_t>(cfg.eval.parent_louds_rank_words_per_super_log2),
                            cfg.eval.parent_louds_build_indices,
                            false,
                            use_norm2_lut,
                            use_norm2_lut,
                            static_cast<int>(norm2_lut_h),
                            25,
                            false,
                            err)) {
        return false;
    }
    if (use_norm2_lut) {
        std::vector<float> centers;
        std::string local_err;
        if (!ReadBinaryFileExact<float>(norm2_lut_centers_f32, static_cast<std::size_t>(norm2_lut_h), &centers, &local_err)) {
            if (err) *err = local_err;
            return false;
        }
        out->provider.SetGlobalNorm2LutCenters(std::move(centers));
    }
    out->provider.SetNorm2DiskLazyEnabled(has_norm2_direct_f32 || use_norm2_lut);

    out->C_root0 = &train.C_root.books.front();
    std::vector<const stlq::ColMajorMatrix<float>*> root_small_books;
    root_small_books.reserve(static_cast<std::size_t>(std::max(0, cfg.model.m - 1)));
    for (int l = 1; l < cfg.model.m; ++l) root_small_books.push_back(&train.C_root.books[static_cast<std::size_t>(l)]);
    out->meta_root_small = stlq::BuildCodebookMeta(root_small_books);
    out->meta_one = stlq::BuildCodebookMeta(stlq::GatherBooks(train.C_one));
    out->cv_by_cid.assign(static_cast<std::size_t>(linkage_list.nlist()), stlq::eval::ClusterView{});
    out->cv_loaded.assign(static_cast<std::size_t>(linkage_list.nlist()), 0);
    return true;
}

bool GetClusterViewCached(ReconContext* ctx, int cid, const stlq::eval::ClusterView** out, std::string* err) {
    if (!ctx || !out) return false;
    *out = nullptr;
    if (cid < 0 || cid >= static_cast<int>(ctx->cv_by_cid.size())) return false;
    if (!ctx->cv_loaded[static_cast<std::size_t>(cid)]) {
        stlq::eval::ClusterProvider::PrepStats stats{};
        if (!ctx->provider.GetCluster(cid, &ctx->cv_by_cid[static_cast<std::size_t>(cid)], &stats, err)) return false;
        ctx->cv_loaded[static_cast<std::size_t>(cid)] = 1;
    }
    *out = &ctx->cv_by_cid[static_cast<std::size_t>(cid)];
    return true;
}

int FindRank(const stlq::ColMajorMatrix<int>& indices, int q, int id) {
    if (id < 0 || q < 0 || q >= indices.cols) return -1;
    const int* row = indices.Col(q);
    for (int r = 0; r < indices.rows; ++r) {
        if (row[r] == id) return r + 1;
    }
    return -1;
}

bool ExportCandidateCsv(const Args& args,
                        const std::filesystem::path& run_root,
                        const stlq::Config& cfg,
                        const stlq::TrainResult& train,
                        const stlq::io::LinkageListReader& linkage_list,
                        const stlq::Dataset& query_dataset,
                        const stlq::RecallResult& adc,
                        const std::unordered_map<std::uint32_t, NodeFeature>& features,
                        std::string* err) {
    const std::filesystem::path out_path(args.out_csv);
    if (out_path.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(out_path.parent_path(), ec);
        if (ec) {
            if (err) *err = "failed to create output directory: " + out_path.parent_path().string();
            return false;
        }
    }
    std::ofstream out(out_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        if (err) *err = "failed to open output CSV: " + out_path.string();
        return false;
    }

    stlq::io::DatasetVectorReader base_reader;
    if (!stlq::io::OpenDatasetVectorReader(cfg, stlq::io::DatasetRole::kBase, &base_reader, err)) return false;
    if (base_reader.d != query_dataset.Xq.rows) {
        if (err) *err = "base/query dimension mismatch";
        return false;
    }

    ReconContext recon;
    if (!OpenReconContext(run_root, cfg, train, linkage_list, &recon, err)) return false;

    out << "query_id,rank,candidate_id,adc_distance,exact_distance,adc_minus_exact,"
        << "query_norm2,raw_norm2,q_dot_raw,"
        << "center_norm2,q_dot_center,raw_dot_center,q_center_distance,raw_center_distance,"
        << "recon_norm2,q_dot_recon,raw_dot_recon,recon_distance,recon_minus_exact,residual_norm2,"
        << "exact_rank_in_shortlist,is_gt1,gt1_id,gt1_adc_rank,gt1_in_shortlist,"
        << "cid,local_id,real_pos,depth,parent_local_id,parent_kind,parent_depth,"
        << "child_degree,subtree_real,cluster_n_real,cluster_n_virtual,cluster_linked_ratio,"
        << "cluster_mean_depth,cluster_max_depth,cluster_max_child_degree,cluster_max_subtree_real,cluster_degree_gini\n";
    out << std::setprecision(10);

    const int k = adc.indices.rows;
    const int nq = adc.indices.cols;
    const int d = query_dataset.Xq.rows;
    for (int q = 0; q < nq; ++q) {
        std::vector<std::uint32_t> ids;
        ids.reserve(static_cast<std::size_t>(k));
        std::vector<int> id_by_rank(static_cast<std::size_t>(k), -1);
        for (int r = 0; r < k; ++r) {
            const int id = adc.indices(r, q);
            id_by_rank[static_cast<std::size_t>(r)] = id;
            if (id >= 0) ids.push_back(static_cast<std::uint32_t>(id));
        }

        stlq::ColMajorMatrix<float> Xb;
        if (!stlq::io::ReadDatasetVectorByIdsF32(base_reader, ids, &Xb, err)) return false;
        if (!stlq::IsIdentityRotation(train.R)) stlq::ApplyRotationInPlace(train.R, &Xb);

        std::vector<double> exact_by_rank(static_cast<std::size_t>(k), std::numeric_limits<double>::quiet_NaN());
        std::vector<double> raw_norm2_by_rank(static_cast<std::size_t>(k), std::numeric_limits<double>::quiet_NaN());
        std::vector<double> q_dot_raw_by_rank(static_cast<std::size_t>(k), std::numeric_limits<double>::quiet_NaN());
        std::vector<double> center_norm2_by_rank(static_cast<std::size_t>(k), std::numeric_limits<double>::quiet_NaN());
        std::vector<double> q_dot_center_by_rank(static_cast<std::size_t>(k), std::numeric_limits<double>::quiet_NaN());
        std::vector<double> raw_dot_center_by_rank(static_cast<std::size_t>(k), std::numeric_limits<double>::quiet_NaN());
        std::vector<double> q_center_distance_by_rank(static_cast<std::size_t>(k), std::numeric_limits<double>::quiet_NaN());
        std::vector<double> raw_center_distance_by_rank(static_cast<std::size_t>(k), std::numeric_limits<double>::quiet_NaN());
        std::vector<double> recon_norm2_by_rank(static_cast<std::size_t>(k), std::numeric_limits<double>::quiet_NaN());
        std::vector<double> q_dot_recon_by_rank(static_cast<std::size_t>(k), std::numeric_limits<double>::quiet_NaN());
        std::vector<double> raw_dot_recon_by_rank(static_cast<std::size_t>(k), std::numeric_limits<double>::quiet_NaN());
        std::vector<double> recon_distance_by_rank(static_cast<std::size_t>(k), std::numeric_limits<double>::quiet_NaN());
        std::vector<double> recon_minus_exact_by_rank(static_cast<std::size_t>(k), std::numeric_limits<double>::quiet_NaN());
        std::vector<double> residual_norm2_by_rank(static_cast<std::size_t>(k), std::numeric_limits<double>::quiet_NaN());
        std::vector<std::pair<double, int>> exact_sorted;
        exact_sorted.reserve(ids.size());
        const float* qv = query_dataset.Xq.Col(q);
        const double q_norm2 = Norm2(qv, d);
        for (int r = 0, valid_col = 0; r < k; ++r) {
            const int id = id_by_rank[static_cast<std::size_t>(r)];
            if (id < 0) continue;
            const float* xb = Xb.Col(valid_col);
            const double raw_norm2 = Norm2(xb, d);
            const double q_dot_raw = Dot(qv, xb, d);
            const double exact = q_norm2 + raw_norm2 - 2.0 * q_dot_raw;
            exact_by_rank[static_cast<std::size_t>(r)] = exact;
            raw_norm2_by_rank[static_cast<std::size_t>(r)] = raw_norm2;
            q_dot_raw_by_rank[static_cast<std::size_t>(r)] = q_dot_raw;

            const auto it = features.find(static_cast<std::uint32_t>(id));
            if (it != features.end() && it->second.cid >= 0 && it->second.real_pos >= 0) {
                if (!train.C_root.books.empty() && it->second.cid < train.C_root.books.front().cols) {
                    const float* center = train.C_root.books.front().Col(it->second.cid);
                    const double center_norm2 = Norm2(center, d);
                    const double q_dot_center = Dot(qv, center, d);
                    const double raw_dot_center = Dot(xb, center, d);
                    center_norm2_by_rank[static_cast<std::size_t>(r)] = center_norm2;
                    q_dot_center_by_rank[static_cast<std::size_t>(r)] = q_dot_center;
                    raw_dot_center_by_rank[static_cast<std::size_t>(r)] = raw_dot_center;
                    q_center_distance_by_rank[static_cast<std::size_t>(r)] = q_norm2 + center_norm2 - 2.0 * q_dot_center;
                    raw_center_distance_by_rank[static_cast<std::size_t>(r)] = raw_norm2 + center_norm2 - 2.0 * raw_dot_center;
                }
                const stlq::eval::ClusterView* cv = nullptr;
                std::string local_err;
                std::vector<float> xhat;
                if (GetClusterViewCached(&recon, it->second.cid, &cv, &local_err) &&
                    cv &&
                    ReconstructRealNodeCoeffCodec(*cv,
                                                  it->second.cid,
                                                  it->second.real_pos,
                                                  *recon.C_root0,
                                                  recon.meta_root_small,
                                                  recon.meta_one,
                                                  &xhat)) {
                    const double recon_norm2 = Norm2(xhat.data(), d);
                    const double q_dot_recon = Dot(qv, xhat.data(), d);
                    const double raw_dot_recon = Dot(xb, xhat.data(), d);
                    const double recon_distance = q_norm2 + recon_norm2 - 2.0 * q_dot_recon;
                    const double residual_norm2 = raw_norm2 + recon_norm2 - 2.0 * raw_dot_recon;
                    recon_norm2_by_rank[static_cast<std::size_t>(r)] = recon_norm2;
                    q_dot_recon_by_rank[static_cast<std::size_t>(r)] = q_dot_recon;
                    raw_dot_recon_by_rank[static_cast<std::size_t>(r)] = raw_dot_recon;
                    recon_distance_by_rank[static_cast<std::size_t>(r)] = recon_distance;
                    recon_minus_exact_by_rank[static_cast<std::size_t>(r)] = recon_distance - exact;
                    residual_norm2_by_rank[static_cast<std::size_t>(r)] = residual_norm2;
                }
            }
            exact_sorted.push_back({exact, r});
            ++valid_col;
        }
        std::sort(exact_sorted.begin(), exact_sorted.end(), [](const auto& a, const auto& b) {
            if (a.first != b.first) return a.first < b.first;
            return a.second < b.second;
        });
        std::vector<int> exact_rank_by_adc_rank(static_cast<std::size_t>(k), -1);
        for (int i = 0; i < static_cast<int>(exact_sorted.size()); ++i) {
            exact_rank_by_adc_rank[static_cast<std::size_t>(exact_sorted[static_cast<std::size_t>(i)].second)] = i + 1;
        }

        const int gt1 = (q < static_cast<int>(query_dataset.gt.size())) ? query_dataset.gt[static_cast<std::size_t>(q)] : -1;
        const int gt_adc_rank = FindRank(adc.indices, q, gt1);
        const int gt_in_shortlist = (gt_adc_rank > 0) ? 1 : 0;

        for (int r = 0; r < k; ++r) {
            const int id = id_by_rank[static_cast<std::size_t>(r)];
            const double adc_dist = adc.dists(r, q);
            const double exact = exact_by_rank[static_cast<std::size_t>(r)];
            const auto it = (id >= 0) ? features.find(static_cast<std::uint32_t>(id)) : features.end();
            const NodeFeature missing;
            const NodeFeature& f = (it == features.end()) ? missing : it->second;
            out << q << ","
                << (r + 1) << ","
                << id << ","
                << adc_dist << ","
                << exact << ","
                << (adc_dist - exact) << ","
                << q_norm2 << ","
                << raw_norm2_by_rank[static_cast<std::size_t>(r)] << ","
                << q_dot_raw_by_rank[static_cast<std::size_t>(r)] << ","
                << center_norm2_by_rank[static_cast<std::size_t>(r)] << ","
                << q_dot_center_by_rank[static_cast<std::size_t>(r)] << ","
                << raw_dot_center_by_rank[static_cast<std::size_t>(r)] << ","
                << q_center_distance_by_rank[static_cast<std::size_t>(r)] << ","
                << raw_center_distance_by_rank[static_cast<std::size_t>(r)] << ","
                << recon_norm2_by_rank[static_cast<std::size_t>(r)] << ","
                << q_dot_recon_by_rank[static_cast<std::size_t>(r)] << ","
                << raw_dot_recon_by_rank[static_cast<std::size_t>(r)] << ","
                << recon_distance_by_rank[static_cast<std::size_t>(r)] << ","
                << recon_minus_exact_by_rank[static_cast<std::size_t>(r)] << ","
                << residual_norm2_by_rank[static_cast<std::size_t>(r)] << ","
                << exact_rank_by_adc_rank[static_cast<std::size_t>(r)] << ","
                << ((id == gt1) ? 1 : 0) << ","
                << gt1 << ","
                << gt_adc_rank << ","
                << gt_in_shortlist << ","
                << f.cid << ","
                << f.local_id << ","
                << f.real_pos << ","
                << f.depth << ","
                << f.parent_local_id << ","
                << ParentKindName(f.parent_kind) << ","
                << f.parent_depth << ","
                << f.child_degree << ","
                << f.subtree_real << ","
                << f.cluster_n_real << ","
                << f.cluster_n_virtual << ","
                << f.cluster_linked_ratio << ","
                << f.cluster_mean_depth << ","
                << f.cluster_max_depth << ","
                << f.cluster_max_child_degree << ","
                << f.cluster_max_subtree_real << ","
                << f.cluster_degree_gini << "\n";
        }
    }
    if (!out) {
        if (err) *err = "failed while writing CSV: " + out_path.string();
        return false;
    }
    return true;
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

    Config cfg;
    TrainResult train;
    Dataset query_dataset;
    io::LinkageListReader linkage_list;
    RecallResult adc;
    if (!RunLinkageEval(args, run_root, cfg_path, &cfg, &train, &query_dataset, &linkage_list, &adc, &err)) {
        LogError(err.empty() ? "linkage eval failed" : err);
        return 1;
    }

    std::unordered_map<std::uint32_t, NodeFeature> features;
    if (!BuildNodeFeatureMap(linkage_list, &features, &err)) {
        LogError(err.empty() ? "BuildNodeFeatureMap failed" : err);
        return 1;
    }
    if (!ExportCandidateCsv(args, run_root, cfg, train, linkage_list, query_dataset, adc, features, &err)) {
        LogError(err.empty() ? "ExportCandidateCsv failed" : err);
        return 1;
    }

    LogInfo("Exported STLQ candidate features: " + args.out_csv);
    LogInfo("Rows: " + std::to_string(static_cast<std::uint64_t>(adc.indices.rows) *
                                      static_cast<std::uint64_t>(adc.indices.cols)));
    return 0;
}
