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
#include "stlq/eval/linkage_cluster_provider.h"
#include "stlq/io/dataset_io.h"
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
        if ((a == '"' && b == '"') || (a == '\'' && b == '\'')) {
            return s.substr(1, s.size() - 2);
        }
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
        } else if (a == "-h" || a == "--help") {
            std::cout
                << "export_stlq_cluster_angles\n\n"
                << "Required:\n"
                << "  --run_root <dir>      STLQ run root containing config_snapshot.txt and linkage_list/\n\n"
                << "Optional:\n"
                << "  --cluster_id <int>    Cluster id to export (default: 0)\n"
                << "  --out <csv>           Output CSV (default: <run_root>/analysis/cluster_<cid>_angles.csv)\n\n"
                << "CSV columns:\n"
                << "  raw_angle_deg,recon_angle_deg\n";
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
    if (out->cluster_id < 0) {
        if (err) *err = "--cluster_id must be non-negative";
        return false;
    }
    if (out->out_csv.empty()) {
        out->out_csv = (std::filesystem::path(out->run_root) / "analysis" /
                        ("cluster_" + std::to_string(out->cluster_id) + "_angles.csv"))
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
        if (line.rfind(prefix, 0) == 0) {
            return StripMatchingQuotes(line.substr(prefix.size()));
        }
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
    if (base_path.empty() || PathExistsForLoadMode(base_path, load_date, load_seq)) {
        return base_path;
    }
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

inline void AddScaledVector(float alpha, const float* src, int d, float* dst) {
    for (int r = 0; r < d; ++r) dst[r] += alpha * src[r];
}

inline std::uint32_t ReadParent1BasedAt(const stlq::eval::ClusterView& cv, int pos) {
    if (pos < 0 || pos >= cv.n_real) return 0u;
    if (cv.parent_is_u16) {
        return cv.parent_1based_u16 ? static_cast<std::uint32_t>(cv.parent_1based_u16[pos]) : 0u;
    }
    return cv.parent_1based ? cv.parent_1based[pos] : 0u;
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

double AngleDeg(const float* a, const float* b, int d) {
    long double dot = 0.0;
    long double na = 0.0;
    long double nb = 0.0;
    for (int i = 0; i < d; ++i) {
        const long double x = static_cast<long double>(a[i]);
        const long double y = static_cast<long double>(b[i]);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if (na <= 0.0 || nb <= 0.0) return std::numeric_limits<double>::quiet_NaN();
    long double c = dot / (std::sqrt(na) * std::sqrt(nb));
    c = std::max<long double>(-1.0, std::min<long double>(1.0, c));
    return static_cast<double>(std::acos(c) * 180.0L / 3.141592653589793238462643383279502884L);
}

bool OpenCoeffProvider(const std::filesystem::path& linkage_list_dir,
                       const stlq::Config& cfg,
                       const stlq::io::LinkageListReader& linkage_list,
                       stlq::io::LinkageCoeffCodecReader* coeff_reader,
                       stlq::eval::ClusterProvider* provider,
                       std::string* err) {
    const std::filesystem::path coeff_meta = linkage_list_dir / "coeff_meta.bin";
    if (!std::filesystem::exists(coeff_meta)) {
        if (err) *err = "coeff_meta.bin not found. This exporter currently requires retained coeff codec artifacts.";
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
        std::string local_err;
        if (!ReadBinaryFileExact<float>(norm2_lut_centers_f32,
                                        static_cast<std::size_t>(norm2_lut_h),
                                        &centers,
                                        &local_err)) {
            if (err) *err = local_err;
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
        std::string train_err;
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
        if (!io::LoadTrainResults(base_path, load_date, load_seq, cfg.model.m, &train, &train_err)) {
            LogError(train_err.empty() ? "LoadTrainResults failed." : train_err);
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
    for (int l = 1; l < cfg.model.m; ++l) {
        root_small_books.push_back(&train.C_root.books[static_cast<std::size_t>(l)]);
    }
    const CodebookMeta meta_root_small = BuildCodebookMeta(root_small_books);
    const CodebookMeta meta_one = BuildCodebookMeta(GatherBooks(train.C_one));

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
    if (base_reader.d != d) {
        LogError("Base vector dim does not match codebook dim.");
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
    if (!IsIdentityRotation(train.R)) {
        ApplyRotationInPlace(train.R, &Xraw);
    }

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
    out << "raw_angle_deg,recon_angle_deg\n";
    out << std::setprecision(10);

    const float* center = C_root0.Col(args.cluster_id);
    std::vector<float> xhat;
    int written = 0;
    for (int i = 0; i < cv.n_real; ++i) {
        if (!ReconstructRealNodeCoeffCodec(cv, args.cluster_id, i, C_root0, meta_root_small, meta_one, &xhat)) {
            LogError("Failed to reconstruct local position " + std::to_string(i));
            return 1;
        }
        const double raw_angle = AngleDeg(Xraw.Col(i), center, d);
        const double recon_angle = AngleDeg(xhat.data(), center, d);
        out << raw_angle << "," << recon_angle << "\n";
        ++written;
    }
    if (!out) {
        LogError("Failed while writing output CSV.");
        return 1;
    }

    LogInfo("Exported STLQ cluster angles: cluster_id=" + std::to_string(args.cluster_id) +
            " rows=" + std::to_string(written) + " out=" + out_path.string());
    return 0;
}
