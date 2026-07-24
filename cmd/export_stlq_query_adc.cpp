#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "stlq/common/config.h"
#include "stlq/common/logger.h"
#include "stlq/core/threading.h"
#include "stlq/eval/linkage_cluster_provider.h"
#include "stlq/eval/query_table_builder.h"
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
    int cluster_id = 0;
    int query_id = 0;
    bool rabitq_analysis = false;
};

struct QueryTablesView {
    const float* xCq_root0_col = nullptr;
    const float* xCq_root_small_col = nullptr;
    const float* xCq_one_col = nullptr;
};

std::string TrimAscii(std::string s) {
    auto is_ws = [](unsigned char c) {
        return c == ' ' || c == '\t' || c == '\r' || c == '\n';
    };
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
        } else if (a == "--out" || a == "--out_csv" || a == "--out-csv") {
            if (!take(&out->out_csv)) {
                if (err) *err = "Missing value for " + a;
                return false;
            }
        } else if (a == "--cluster_id" || a == "--cluster-id" || a == "--cid") {
            std::string v;
            if (!take(&v)) {
                if (err) *err = "Missing value for " + a;
                return false;
            }
            out->cluster_id = std::stoi(v);
        } else if (a == "--query_id" || a == "--query-id" || a == "--qid") {
            std::string v;
            if (!take(&v)) {
                if (err) *err = "Missing value for " + a;
                return false;
            }
            out->query_id = std::stoi(v);
        } else if (a == "--rabitq_analysis" || a == "--rabitq-analysis") {
            out->rabitq_analysis = true;
        } else if (a == "-h" || a == "--help") {
            std::cout
                << "export_stlq_query_adc\n\n"
                << "Required:\n"
                << "  --run_root <dir>      STLQ run root containing config_snapshot.txt and linkage_list/\n\n"
                << "Optional:\n"
                << "  --cluster_id <int>    Cluster id to export (default: 0)\n"
                << "  --query_id <int>      Query id to use (default: 0)\n"
                << "  --rabitq_analysis     Add RaBitQ-style bias/projection diagnostic columns\n"
                << "  --out <csv>           Output CSV (default: <run_root>/analysis/query_<qid>_cluster_<cid>_adc.csv)\n\n"
                << "CSV columns:\n"
                << "  adc_distance,true_distance\n"
                << "  with --rabitq_analysis: local_pos,base_id,adc_distance,true_distance,"
                << "residual_norm2,gamma,z_norm2,d1_residual_corrected,d2_projection_corrected,"
                << "directional_term,q_norm2,x_norm2,xhat_norm2,q_dot_x,q_dot_xhat,x_dot_xhat\n";
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
    if (out->cluster_id < 0 || out->query_id < 0) {
        if (err) *err = "--cluster_id and --query_id must be non-negative";
        return false;
    }
    if (out->out_csv.empty()) {
        out->out_csv = (std::filesystem::path(out->run_root) / "analysis" /
                        ("query_" + std::to_string(out->query_id) +
                         "_cluster_" + std::to_string(out->cluster_id) + "_adc.csv"))
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

template <typename T>
bool ReadBinaryFileExact(const std::filesystem::path& path, std::size_t count, std::vector<T>* out, std::string* err) {
    if (!out) return false;
    out->clear();
    if (count == 0) return true;
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        if (err) *err = "Failed to open: " + path.string();
        return false;
    }
    out->resize(count);
    in.read(reinterpret_cast<char*>(out->data()), static_cast<std::streamsize>(count * sizeof(T)));
    if (!in || static_cast<std::size_t>(in.gcount()) != count * sizeof(T)) {
        if (err) *err = "Failed to read: " + path.string();
        return false;
    }
    return true;
}

inline std::uint32_t ReadParent1BasedAt(const stlq::eval::ClusterView& cv, int pos) {
    if (pos < 0 || pos >= cv.n_real) return 0u;
    if (cv.parent_is_u16) {
        return cv.parent_1based_u16 ? static_cast<std::uint32_t>(cv.parent_1based_u16[pos]) : 0u;
    }
    return cv.parent_1based ? cv.parent_1based[pos] : 0u;
}

float DotHatRealByPathCoeffCodec(const stlq::eval::ClusterView& cv,
                                 int cid,
                                 int local_pos,
                                 const std::vector<int>& offsets_root_small,
                                 const std::vector<int>& offsets_one,
                                 const QueryTablesView& qv) {
    if (local_pos < 0 || local_pos >= cv.n_real) return std::numeric_limits<float>::quiet_NaN();
    if (!cv.q_layer_major || !cv.scales_root || !cv.scales_linkage) return std::numeric_limits<float>::quiet_NaN();
    if (!cv.codes_small_bytes || !cv.code0_one_bytes) return std::numeric_limits<float>::quiet_NaN();
    if (cv.n_virt > 0 && !cv.virt_codes_small_bytes) return std::numeric_limits<float>::quiet_NaN();

    const int m = cv.m;
    const int m_codes = cv.m_codes;
    const int n_virt = cv.n_virt;
    const int nc = cv.nc;
    if (m <= 1 || m_codes != std::max(0, m - 1) || nc != cv.n_real + cv.n_virt) {
        return std::numeric_limits<float>::quiet_NaN();
    }

    int local = n_virt + local_pos;
    std::vector<int> stack;
    stack.reserve(32);
    while (true) {
        if (local < n_virt) break;
        const int pos = local - n_virt;
        if (pos < 0 || pos >= cv.n_real) return std::numeric_limits<float>::quiet_NaN();
        if (pos < cv.n_root_real) break;
        stack.push_back(local);
        const std::uint32_t p1 = ReadParent1BasedAt(cv, pos);
        if (p1 == 0u) break;
        local = static_cast<int>(p1) - 1;
        if (local < 0 || local >= nc) return std::numeric_limits<float>::quiet_NaN();
    }

    auto q_at = [&](int layer, int local_idx) -> float {
        return static_cast<float>(cv.q_layer_major[static_cast<std::size_t>(layer) * static_cast<std::size_t>(nc) +
                                                   static_cast<std::size_t>(local_idx)]);
    };

    float dot = cv.scales_root[0] * q_at(0, local) * qv.xCq_root0_col[cid];
    for (int l = 1; l < m; ++l) {
        const int off = offsets_root_small[static_cast<std::size_t>(l)];
        const std::uint8_t code = (local < n_virt)
            ? cv.virt_codes_small_bytes[static_cast<std::size_t>(local) * static_cast<std::size_t>(m_codes) +
                                        static_cast<std::size_t>(l - 1)]
            : cv.codes_small_bytes[static_cast<std::size_t>(local - n_virt) * static_cast<std::size_t>(m_codes) +
                                   static_cast<std::size_t>(l - 1)];
        dot += cv.scales_root[l] * q_at(l, local) * qv.xCq_root_small_col[off + static_cast<int>(code)];
    }

    while (!stack.empty()) {
        const int child_local = stack.back();
        stack.pop_back();
        const int pos = child_local - n_virt;
        if (pos < 0 || pos >= cv.n_real) return std::numeric_limits<float>::quiet_NaN();
        const std::uint8_t code0 = cv.code0_one_bytes[pos];
        dot += cv.scales_linkage[0] * q_at(0, child_local) *
               qv.xCq_one_col[offsets_one[0] + static_cast<int>(code0)];
        const std::uint8_t* row =
            cv.codes_small_bytes + static_cast<std::size_t>(pos) * static_cast<std::size_t>(m_codes);
        for (int l = 1; l < m; ++l) {
            const std::uint8_t code = row[l - 1];
            dot += cv.scales_linkage[l] * q_at(l, child_local) *
                   qv.xCq_one_col[offsets_one[static_cast<std::size_t>(l)] + static_cast<int>(code)];
        }
    }
    return dot;
}

double Norm2(const float* x, int d) {
    long double s = 0.0;
    for (int i = 0; i < d; ++i) {
        const long double v = static_cast<long double>(x[i]);
        s += v * v;
    }
    return static_cast<double>(s);
}

double Dot(const float* a, const float* b, int d) {
    long double s = 0.0;
    for (int i = 0; i < d; ++i) {
        s += static_cast<long double>(a[i]) * static_cast<long double>(b[i]);
    }
    return static_cast<double>(s);
}

double SquaredDistance(const float* a, const float* b, int d) {
    long double s = 0.0;
    for (int i = 0; i < d; ++i) {
        const long double diff = static_cast<long double>(a[i]) - static_cast<long double>(b[i]);
        s += diff * diff;
    }
    return static_cast<double>(s);
}

inline void AddScaledVector(float alpha, const float* src, int d, float* dst) {
    for (int r = 0; r < d; ++r) dst[r] += alpha * src[r];
}

float RuntimeNormHat(const stlq::eval::ClusterView& cv, int local_pos) {
    if (local_pos < 0 || local_pos >= cv.n_real) return std::numeric_limits<float>::quiet_NaN();
    if (cv.r_norm2) return cv.r_norm2[local_pos];
    if (cv.r_norm2_lut_u8 && cv.r_norm2_lut_centers && cv.r_norm2_lut_size > 0) {
        const std::uint8_t c = cv.r_norm2_lut_u8[local_pos];
        return cv.r_norm2_lut_centers[std::min<int>(static_cast<int>(c), cv.r_norm2_lut_size - 1)];
    }
    return std::numeric_limits<float>::quiet_NaN();
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

    const float a0 = cv.scales_root[0] * q_i8(0, root_local);
    AddScaledVector(a0, C_root0.Col(cid), d, xhat);
    for (int l = 1; l < m; ++l) {
        const int off = meta_root_small.offsets[static_cast<std::size_t>(l - 1)];
        const std::uint8_t code =
            (root_local < n_virt)
                ? cv.virt_codes_small_bytes[static_cast<std::size_t>(root_local) * static_cast<std::size_t>(stride_codes) +
                                            static_cast<std::size_t>(l - 1)]
                : cv.codes_small_bytes[static_cast<std::size_t>(root_local - n_virt) * static_cast<std::size_t>(stride_codes) +
                                       static_cast<std::size_t>(l - 1)];
        const float a = cv.scales_root[l] * q_i8(l, root_local);
        AddScaledVector(a, meta_root_small.flat.Col(off + static_cast<int>(code)), d, xhat);
    }

    while (!linkage_stack.empty()) {
        const int child_local = linkage_stack.back();
        linkage_stack.pop_back();
        const int pos = child_local - n_virt;
        if (pos < 0 || pos >= cv.n_real) return false;

        const std::uint8_t code0 = cv.code0_one_bytes[pos];
        const float b0 = cv.scales_linkage[0] * q_i8(0, child_local);
        AddScaledVector(b0, meta_one.flat.Col(meta_one.offsets[0] + static_cast<int>(code0)), d, xhat);

        const std::uint8_t* row =
            cv.codes_small_bytes + static_cast<std::size_t>(pos) * static_cast<std::size_t>(stride_codes);
        for (int l = 1; l < m; ++l) {
            const std::uint8_t code = row[l - 1];
            const float b = cv.scales_linkage[l] * q_i8(l, child_local);
            AddScaledVector(b,
                            meta_one.flat.Col(meta_one.offsets[static_cast<std::size_t>(l)] + static_cast<int>(code)),
                            d,
                            xhat);
        }
    }
    return true;
}

bool OpenCoeffProvider(const std::filesystem::path& linkage_list_dir,
                       const stlq::Config& cfg,
                       const stlq::io::LinkageListReader& linkage_list,
                       stlq::io::LinkageCoeffCodecReader* coeff_reader,
                       stlq::eval::ClusterProvider* provider,
                       std::string* err) {
    const std::filesystem::path coeff_meta = linkage_list_dir / "coeff_meta.bin";
    if (!std::filesystem::exists(coeff_meta)) {
        if (err) *err = "coeff_meta.bin not found. This exporter requires retained coeff codec artifacts.";
        return false;
    }
    if (!coeff_reader->Open(linkage_list_dir.string(), err)) return false;

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

    if (!provider->Open(linkage_list,
                        coeff_reader,
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
        if (!ReadBinaryFileExact<float>(norm2_lut_centers_f32, static_cast<std::size_t>(norm2_lut_h), &centers, err)) {
            return false;
        }
        provider->SetGlobalNorm2LutCenters(std::move(centers));
    }
    provider->SetNorm2DiskLazyEnabled(has_norm2_direct_f32 || use_norm2_lut);
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

    Config cfg = DefaultConfig(true);
    NormalizeHVec(&cfg);
    std::vector<int> ignored;
    if (!LoadConfigFile(cfg_path.string(), &cfg, &err, &ignored)) {
        LogError(err);
        return 1;
    }

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

    if (train.C_root.books.empty() || train.C_one.books.empty()) {
        LogError("TrainResult missing C_root/C_one.");
        return 1;
    }
    const ColMajorMatrix<float>& C_root0 = train.C_root.books.front();
    const int d = C_root0.rows;
    if (args.cluster_id >= C_root0.cols) {
        LogError("--cluster_id exceeds C_root[0] nlist.");
        return 1;
    }

    std::vector<const ColMajorMatrix<float>*> root_small_books;
    root_small_books.reserve(static_cast<std::size_t>(std::max(0, cfg.model.m - 1)));
    for (int l = 1; l < cfg.model.m; ++l) root_small_books.push_back(&train.C_root.books[static_cast<std::size_t>(l)]);
    const CodebookMeta meta_root_small = BuildCodebookMeta(root_small_books);
    const CodebookMeta meta_one = BuildCodebookMeta(GatherBooks(train.C_one));

    ColMajorMatrix<float> Xq;
    std::vector<int> gt_ignored;
    if (!io::LoadQuerySet(cfg, &Xq, &gt_ignored, &err)) {
        LogError(err);
        return 1;
    }
    if (Xq.rows != d || args.query_id >= Xq.cols) {
        LogError("--query_id exceeds query count or query dimension mismatches codebooks.");
        return 1;
    }
    if (!IsIdentityRotation(train.R)) ApplyRotationInPlace(train.R, &Xq);
    const float* q = Xq.Col(args.query_id);
    const double qnorm2 = Norm2(q, d);

    const std::filesystem::path linkage_list_dir = run_root / "linkage_list";
    io::LinkageListReader linkage_list;
    if (!linkage_list.Open(linkage_list_dir.string(), &err)) {
        LogError(err);
        return 1;
    }
    if (args.cluster_id >= linkage_list.nlist()) {
        LogError("--cluster_id exceeds linkage_list nlist.");
        return 1;
    }

    io::LinkageCoeffCodecReader coeff_reader;
    eval::ClusterProvider provider;
    if (!OpenCoeffProvider(linkage_list_dir, cfg, linkage_list, &coeff_reader, &provider, &err)) {
        LogError(err);
        return 1;
    }
    eval::ClusterView cv;
    eval::ClusterProvider::PrepStats prep{};
    if (!provider.GetCluster(args.cluster_id, &cv, &prep, &err)) {
        LogError(err.empty() ? "ClusterProvider.GetCluster failed." : err);
        return 1;
    }
    if (cv.n_real <= 0) {
        LogError("Selected cluster is empty.");
        return 1;
    }

    io::DatasetVectorReader base_reader;
    if (!io::OpenDatasetVectorReader(cfg, io::DatasetRole::kBase, &base_reader, &err)) {
        LogError(err);
        return 1;
    }
    std::vector<std::uint32_t> ids(static_cast<std::size_t>(cv.n_real));
    for (int i = 0; i < cv.n_real; ++i) ids[static_cast<std::size_t>(i)] = cv.real_ids[i];
    ColMajorMatrix<float> Xraw;
    if (!io::ReadDatasetVectorByIdsF32(base_reader, ids, &Xraw, &err)) {
        LogError(err);
        return 1;
    }
    if (Xraw.rows != d || Xraw.cols != cv.n_real) {
        LogError("Loaded raw vector matrix shape mismatch.");
        return 1;
    }
    if (!IsIdentityRotation(train.R)) ApplyRotationInPlace(train.R, &Xraw);

    std::vector<int> offsets_root_small(static_cast<std::size_t>(cfg.model.m), 0);
    for (int l = 1; l < cfg.model.m; ++l) {
        offsets_root_small[static_cast<std::size_t>(l)] = meta_root_small.offsets[static_cast<std::size_t>(l - 1)];
    }
    double t0 = 0.0, t1 = 0.0, t2 = 0.0;
    STLQueryTables qt = BuildSTLQueryTables(q,
                                            d,
                                            d,
                                            1,
                                            C_root0,
                                            meta_root_small,
                                            meta_one,
                                            std::max(1, OmpMaxThreads()),
                                            &t0,
                                            &t1,
                                            &t2);
    (void)t0;
    (void)t1;
    (void)t2;
    QueryTablesView qv;
    qv.xCq_root0_col = qt.xCq_root0.Col(0);
    qv.xCq_root_small_col = qt.xCq_root_small.Col(0);
    qv.xCq_one_col = qt.xCq_one.Col(0);

    const std::filesystem::path out_path(args.out_csv);
    if (out_path.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(out_path.parent_path(), ec);
        if (ec) {
            LogError("Failed to create output directory: " + out_path.parent_path().string());
            return 1;
        }
    }
    std::ofstream out(out_path, std::ios::trunc);
    if (!out.is_open()) {
        LogError("Failed to open output CSV: " + out_path.string());
        return 1;
    }
    if (args.rabitq_analysis) {
        out << "local_pos,base_id,adc_distance,true_distance,residual_norm2,gamma,z_norm2,"
               "d1_residual_corrected,d2_projection_corrected,directional_term,"
               "q_norm2,x_norm2,xhat_norm2,q_dot_x,q_dot_xhat,x_dot_xhat\n";
    } else {
        out << "adc_distance,true_distance\n";
    }
    out << std::setprecision(10);

    int written = 0;
    std::vector<float> xhat;
    for (int i = 0; i < cv.n_real; ++i) {
        const float dot_hat = DotHatRealByPathCoeffCodec(cv, args.cluster_id, i, offsets_root_small, meta_one.offsets, qv);
        const float norm_hat = RuntimeNormHat(cv, i);
        if (!std::isfinite(dot_hat) || !std::isfinite(norm_hat)) {
            LogError("Failed to compute ADC terms for local position " + std::to_string(i));
            return 1;
        }
        const double adc_distance = qnorm2 + static_cast<double>(norm_hat) - 2.0 * static_cast<double>(dot_hat);
        const double true_distance = SquaredDistance(q, Xraw.Col(i), d);
        if (args.rabitq_analysis) {
            if (!ReconstructRealNodeCoeffCodec(cv, args.cluster_id, i, C_root0, meta_root_small, meta_one, &xhat)) {
                LogError("Failed to reconstruct local position " + std::to_string(i));
                return 1;
            }
            const float* x = Xraw.Col(i);
            const double x_norm2 = Norm2(x, d);
            const double xhat_norm2 = Norm2(xhat.data(), d);
            const double q_dot_x = Dot(q, x, d);
            const double q_dot_xhat = Dot(q, xhat.data(), d);
            const double x_dot_xhat = Dot(x, xhat.data(), d);
            const double residual_norm2 = std::max(0.0, x_norm2 + xhat_norm2 - 2.0 * x_dot_xhat);
            const double gamma = (x_norm2 > 0.0) ? (x_dot_xhat / x_norm2) : std::numeric_limits<double>::quiet_NaN();
            const double z_norm2 = std::isfinite(gamma)
                                       ? std::max(0.0, xhat_norm2 - gamma * gamma * x_norm2)
                                       : std::numeric_limits<double>::quiet_NaN();
            const double d1 = adc_distance - residual_norm2;
            const double d2 = (std::isfinite(gamma) && std::abs(gamma) > 1e-12)
                                  ? (qnorm2 + x_norm2 - 2.0 * q_dot_xhat / gamma)
                                  : std::numeric_limits<double>::quiet_NaN();
            const double directional_term = 0.5 * (d1 - true_distance);
            out << i << "," << cv.real_ids[i] << ","
                << adc_distance << "," << true_distance << ","
                << residual_norm2 << "," << gamma << "," << z_norm2 << ","
                << d1 << "," << d2 << "," << directional_term << ","
                << qnorm2 << "," << x_norm2 << "," << xhat_norm2 << ","
                << q_dot_x << "," << q_dot_xhat << "," << x_dot_xhat << "\n";
        } else {
            out << adc_distance << "," << true_distance << "\n";
        }
        ++written;
    }
    if (!out) {
        LogError("Failed while writing output CSV.");
        return 1;
    }
    LogInfo("Exported STLQ query ADC distances: query_id=" + std::to_string(args.query_id) +
            " cluster_id=" + std::to_string(args.cluster_id) +
            (args.rabitq_analysis ? " rabitq_analysis=1" : "") +
            " rows=" + std::to_string(written) + " out=" + out_path.string());
    return 0;
}
