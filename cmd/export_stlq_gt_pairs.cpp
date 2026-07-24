#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <omp.h>

#include "stlq/pipeline/app_utils.h"
#include "stlq/common/config.h"
#include "stlq/core/threading.h"
#include "stlq/eval/linkage_cluster_provider.h"
#include "stlq/eval/query_table_builder.h"
#include "stlq/io/linkage_coeff_codec_store.h"
#include "stlq/io/linkage_list_store.h"
#include "stlq/io/dataset_io.h"
#include "stlq/io/dataset_reader_factory.h"
#include "stlq/io/result_io.h"
#include "stlq/common/logger.h"
#include "stlq/quantizer/codebook_meta.h"
#include "stlq/succinct/parent_louds.h"
#include "stlq/common/timer.h"

namespace {

struct Args {
    std::string run_root;
    std::string out_dir;
    int gt_topL = 100;
    int q_block = 64;
};

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
        auto take_int = [&](int* dst) -> bool {
            if (i + 1 >= argc) return false;
            *dst = std::stoi(argv[++i]);
            return true;
        };

        if (a == "--run_root") {
            if (!take(&out->run_root)) {
                if (err) *err = "Missing value for --run_root";
                return false;
            }
        } else if (a == "--out_dir") {
            if (!take(&out->out_dir)) {
                if (err) *err = "Missing value for --out_dir";
                return false;
            }
        } else if (a == "--gt_topL") {
            if (!take_int(&out->gt_topL)) {
                if (err) *err = "Missing value for --gt_topL";
                return false;
            }
        } else if (a == "--q_block") {
            if (!take_int(&out->q_block)) {
                if (err) *err = "Missing value for --q_block";
                return false;
            }
        } else if (a == "-h" || a == "--help") {
            std::cout
                << "export_stlq_gt_pairs\n\n"
                << "Required:\n"
                << "  --run_root <path>   stlq run_root (contains config_snapshot.txt and linkage_list/)\n\n"
                << "Optional:\n"
                << "  --out_dir <path>    Output dir (default: <run_root>/analysis_export)\n"
                << "  --gt_topL <int>     Use top-L groundtruth ids per query (default: 100)\n"
                << "  --q_block <int>     Query block size for GEMM table build (default: 64)\n";
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
    if (out->out_dir.empty()) {
        out->out_dir = (std::filesystem::path(out->run_root) / "analysis_export").string();
    }
    out->gt_topL = std::max(1, out->gt_topL);
    out->q_block = std::max(1, out->q_block);
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

std::string ReadRunStateValue(const std::filesystem::path& run_root, const std::string& key) {
    const std::filesystem::path path = run_root / "run_state.txt";
    std::ifstream in(path);
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
    std::string local_err;
    if (load_date == "raw") {
        std::error_code ec;
        return std::filesystem::exists(base_path, ec) && !ec;
    }
    const std::string dated = stlq::io::MakeDatedPathForLoad(base_path, load_date, load_seq, &local_err);
    return !dated.empty();
}

std::string ResolveLoadBasePath(const std::string& base_path,
                                const std::string& load_date,
                                const std::string& load_seq,
                                const std::filesystem::path& run_root,
                                const std::filesystem::path& cfg_path) {
    if (base_path.empty()) return {};
    if (PathExistsForLoadMode(base_path, load_date, load_seq)) {
        return base_path;
    }

    const std::filesystem::path p(base_path);
    if (p.is_absolute()) {
        return base_path;
    }

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
        if (PathExistsForLoadMode(cand, load_date, load_seq)) {
            return cand;
        }
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

bool WriteBinaryFile(const std::filesystem::path& path, const void* data, std::size_t bytes, std::string* err) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        if (err) *err = "Failed to write: " + path.string();
        return false;
    }
    if (bytes) {
        out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    }
    if (!out) {
        if (err) *err = "Failed to write: " + path.string();
        return false;
    }
    return true;
}

inline float DotF32(const float* a, const float* b, int d) {
    double acc = 0.0;
    for (int i = 0; i < d; ++i) {
        acc += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return static_cast<float>(acc);
}

inline float Norm2F32(const float* a, int d) {
    double acc = 0.0;
    for (int i = 0; i < d; ++i) {
        const double v = static_cast<double>(a[i]);
        acc += v * v;
    }
    return static_cast<float>(acc);
}

int ReadCode0One(const std::uint8_t* base, int pos, int width_bytes) {
    if (!base || pos < 0) return 0;
    if (width_bytes == 1) {
        return static_cast<int>(base[pos]);
    }
    if (width_bytes == 2) {
        std::uint16_t v;
        std::memcpy(&v, base + static_cast<std::size_t>(pos) * 2u, sizeof(v));
        return static_cast<int>(v);
    }
    if (width_bytes == 4) {
        std::uint32_t v;
        std::memcpy(&v, base + static_cast<std::size_t>(pos) * 4u, sizeof(v));
        return static_cast<int>(v);
    }
    return static_cast<int>(base[pos]);
}

struct ClusterCache {
    bool loaded = false;
    std::uint64_t real_lo = 0;
    std::uint64_t virt_lo = 0;
    int n_real = 0;
    int n_virt = 0;
    int n_root_real = 0;
    std::vector<std::uint32_t> parent_real_1based;  // length n_real; values are local 1-based in [0, n_real+n_virt]
};

[[maybe_unused]] bool LoadClusterCache(const stlq::io::LinkageListReader& linkage_list,
                     int cid,
                     int parent_louds_rank_words_per_super_log2,
                     bool parent_louds_build_indices,
                     int parent_louds_select_stride,
                     ClusterCache* out,
                     std::string* err) {
    if (!out) return false;

    std::uint64_t real_lo = 0, real_hi = 0;
    std::uint64_t virt_lo = 0, virt_hi = 0;
    std::uint64_t depth_lo = 0, depth_hi = 0;
    if (!linkage_list.ReadClusterSpan(cid, &real_lo, &real_hi, &virt_lo, &virt_hi, &depth_lo, &depth_hi, err)) {
        return false;
    }
    const int n_real = static_cast<int>(real_hi - real_lo);
    const int n_virt = static_cast<int>(virt_hi - virt_lo);
    if (n_real < 0 || n_virt < 0) {
        if (err) *err = "Invalid cluster span.";
        return false;
    }

    std::vector<std::uint32_t> depth_offsets;
    std::uint32_t n_real_check = 0;
    if (!linkage_list.ReadClusterDepthOffsets(cid, &depth_offsets, &n_real_check, err)) {
        return false;
    }
    if (n_real_check != static_cast<std::uint32_t>(n_real)) {
        // Best-effort; keep going.
    }
    int n_root_real = 0;
    if (depth_offsets.size() >= 2) {
        n_root_real = static_cast<int>(depth_offsets[1]);
    } else {
        n_root_real = n_real;
    }

    std::vector<std::uint8_t> blob;
    if (!linkage_list.ReadClusterParentLOUDSBlob(cid, &blob, err) || blob.empty()) {
        if (err && err->empty()) *err = "Missing parent LOUDS blob.";
        return false;
    }

    try {
        stlq::succinct::ParentLOUDS louds;
        louds.Deserialize(blob.data(),
                          blob.size(),
                          static_cast<std::uint32_t>(std::max(1, parent_louds_rank_words_per_super_log2)),
                          parent_louds_build_indices,
                          static_cast<std::uint32_t>(std::max(1, parent_louds_select_stride)));
        const std::size_t nc = static_cast<std::size_t>(n_real) + static_cast<std::size_t>(n_virt);
        std::vector<std::uint32_t> parent_all;
        louds.DecodeParent1Based(&parent_all, nc);

        out->parent_real_1based.resize(static_cast<std::size_t>(n_real));
        for (int i = 0; i < n_real; ++i) {
            out->parent_real_1based[static_cast<std::size_t>(i)] = parent_all[static_cast<std::size_t>(n_virt + i)];
        }
    } catch (...) {
        if (err) *err = "Failed to decode parent LOUDS.";
        return false;
    }

    out->loaded = true;
    out->real_lo = real_lo;
    out->virt_lo = virt_lo;
    out->n_real = n_real;
    out->n_virt = n_virt;
    out->n_root_real = n_root_real;
    return true;
}

struct STLQGlobalArrays {
    int m = 0;
    int m_codes = 0;
    int code0_width_bytes = 1;
    std::uint64_t total_real = 0;
    std::uint64_t total_virt = 0;

    bool has_float_coeffs = false;
    bool has_norm2_direct = false;

    std::vector<std::uint32_t> real_ids;
    std::vector<std::uint8_t> codes_small;   // total_real*m_codes
    std::vector<float> coeffs_small;         // total_real*m_codes
    std::vector<std::uint8_t> code0_one;     // total_real*width
    std::vector<float> a0;                   // total_real

    std::vector<std::uint8_t> virt_codes_small;  // total_virt*m_codes
    std::vector<float> virt_coeffs_small;        // total_virt*m_codes
    std::vector<float> virt_a0;                  // total_virt

    std::vector<std::uint8_t> norm2_codes_u8; // total_real
    std::vector<float> norm2_centers_f32;     // >= max(code)+1

    // Some runs store r_norm2 directly as float (e.g. norm2_int8.f32) instead of a LUT.
    std::vector<float> norm2_direct_f32;  // total_real
};

template <typename T>
bool ReadBinaryFileIfExistsExact(const std::filesystem::path& path,
                                std::size_t count,
                                std::vector<T>* out,
                                std::string* err) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || ec) {
        if (out) out->clear();
        return true;
    }
    return ReadBinaryFileExact<T>(path, count, out, err);
}

bool LoadSTLQGlobalArrays(const std::filesystem::path& linkage_list_dir,
                           const stlq::io::LinkageListReader& linkage_list,
                           int m,
                           bool use_coeff_codec,
                           STLQGlobalArrays* out,
                           std::string* err) {
    if (!out) return false;

    const int m_codes = std::max(0, m - 1);
    const std::uint64_t total_real = linkage_list.total_real();
    const std::uint64_t total_virt = linkage_list.total_virtual();
    const int code0_width = std::max(1, linkage_list.code0_width_bytes());

    *out = {};
    out->m = m;
    out->m_codes = m_codes;
    out->code0_width_bytes = code0_width;
    out->total_real = total_real;
    out->total_virt = total_virt;

    if (linkage_list.m_codes() != m_codes) {
        if (err) {
            *err = "m_codes mismatch: config m=" + std::to_string(m) +
                   " => m_codes=" + std::to_string(m_codes) +
                   " but linkage_list meta.m_codes=" + std::to_string(linkage_list.m_codes());
        }
        return false;
    }

    if (!ReadBinaryFileExact<std::uint32_t>(linkage_list_dir / "real_ids.u32", static_cast<std::size_t>(total_real), &out->real_ids, err)) {
        return false;
    }
    // Optional float-coeff fallback (not present in coeff-codec-only runs).
    // We only enable the fallback if ALL required files exist.
    {
        std::string local_err;
        const bool ok_codes = ReadBinaryFileIfExistsExact<std::uint8_t>(
            linkage_list_dir / "codes.bin",
            static_cast<std::size_t>(total_real) * static_cast<std::size_t>(m_codes),
            &out->codes_small,
            &local_err);
        if (!ok_codes) { if (err) *err = local_err; return false; }

        const bool ok_code0 = ReadBinaryFileIfExistsExact<std::uint8_t>(
            linkage_list_dir / "code0_one.bin",
            static_cast<std::size_t>(total_real) * static_cast<std::size_t>(code0_width),
            &out->code0_one,
            &local_err);
        if (!ok_code0) { if (err) *err = local_err; return false; }

        const bool ok_a0 = ReadBinaryFileIfExistsExact<float>(
            linkage_list_dir / "a0.f32",
            static_cast<std::size_t>(total_real),
            &out->a0,
            &local_err);
        if (!ok_a0) { if (err) *err = local_err; return false; }

        const bool ok_coeffs = ReadBinaryFileIfExistsExact<float>(
            linkage_list_dir / "coeffs.f32",
            static_cast<std::size_t>(total_real) * static_cast<std::size_t>(m_codes),
            &out->coeffs_small,
            &local_err);
        if (!ok_coeffs) { if (err) *err = local_err; return false; }

        const bool ok_vcodes = ReadBinaryFileIfExistsExact<std::uint8_t>(
            linkage_list_dir / "virt_codes.bin",
            static_cast<std::size_t>(total_virt) * static_cast<std::size_t>(m_codes),
            &out->virt_codes_small,
            &local_err);
        if (!ok_vcodes) { if (err) *err = local_err; return false; }

        const bool ok_va0 = ReadBinaryFileIfExistsExact<float>(
            linkage_list_dir / "virt_a0.f32",
            static_cast<std::size_t>(total_virt),
            &out->virt_a0,
            &local_err);
        if (!ok_va0) { if (err) *err = local_err; return false; }

        const bool ok_vcoeffs = ReadBinaryFileIfExistsExact<float>(
            linkage_list_dir / "virt_coeffs.f32",
            static_cast<std::size_t>(total_virt) * static_cast<std::size_t>(m_codes),
            &out->virt_coeffs_small,
            &local_err);
        if (!ok_vcoeffs) { if (err) *err = local_err; return false; }

        out->has_float_coeffs =
            !out->codes_small.empty() && !out->code0_one.empty() && !out->a0.empty() && !out->coeffs_small.empty() &&
            (total_virt == 0 || (!out->virt_codes_small.empty() && !out->virt_a0.empty() && !out->virt_coeffs_small.empty()));
    }

    // Optional norm2 sources:
    // (1) Direct float array (preferred when present): norm2_int8.f32 or norm2_float.f32
    // (2) LUT codes + centers: norm2_*_lut_codes.u8 + norm2_*_lut_centers.f32
    {
        std::string local_err;
        const std::filesystem::path direct_path = use_coeff_codec ? (linkage_list_dir / "norm2_int8.f32")
                                                                  : (linkage_list_dir / "norm2_float.f32");
        if (std::error_code ec; std::filesystem::exists(direct_path, ec) && !ec) {
            if (!ReadBinaryFileExact<float>(direct_path, static_cast<std::size_t>(total_real), &out->norm2_direct_f32, &local_err)) {
                if (err) *err = local_err;
                return false;
            }
            out->has_norm2_direct = true;
        }

        const char* lut_prefix = use_coeff_codec ? "norm2_int8_lut" : "norm2_float_lut";
        const auto codes_path = linkage_list_dir / (std::string(lut_prefix) + "_codes.u8");
        const auto centers_path = linkage_list_dir / (std::string(lut_prefix) + "_centers.f32");

        if (std::error_code ec; std::filesystem::exists(codes_path, ec) && !ec) {
            if (!ReadBinaryFileExact<std::uint8_t>(codes_path,
                                                   static_cast<std::size_t>(total_real),
                                                   &out->norm2_codes_u8,
                                                   &local_err)) {
                if (err) *err = local_err;
                return false;
            }
            // centers length is not known; read by file size.
            std::ifstream in(centers_path, std::ios::binary | std::ios::ate);
            if (!in.is_open()) {
                if (err) *err = "Failed to open: " + centers_path.string();
                return false;
            }
            const auto sz = static_cast<std::size_t>(in.tellg());
            if (sz % sizeof(float) != 0) {
                if (err) *err = "Invalid centers file size: " + centers_path.string();
                return false;
            }
            const std::size_t n = sz / sizeof(float);
            in.seekg(0, std::ios::beg);
            out->norm2_centers_f32.resize(n);
            in.read(reinterpret_cast<char*>(out->norm2_centers_f32.data()), static_cast<std::streamsize>(sz));
            if (!in) {
                if (err) *err = "Failed to read: " + centers_path.string();
                return false;
            }
        }
    }

    return true;
}

struct QueryTablesView {
    const float* xCq_root0_col = nullptr;      // length nlist
    const float* xCq_root_small_col = nullptr; // length Hrs
    const float* xCq_one_col = nullptr;        // length Ho
};

STLQ_ALWAYS_INLINE void AddScaledVectorF32(float alpha, const float* __restrict src, int d, float* __restrict dst) {
    for (int r = 0; r < d; ++r) {
        dst[r] += alpha * src[r];
    }
}

inline std::uint32_t ReadParent1BasedAt(const stlq::eval::ClusterView& cv, int pos) {
    if (pos < 0 || pos >= cv.n_real) {
        return 0u;
    }
    if (cv.parent_is_u16) {
        return cv.parent_1based_u16 ? static_cast<std::uint32_t>(cv.parent_1based_u16[pos]) : 0u;
    }
    return cv.parent_1based ? cv.parent_1based[pos] : 0u;
}

// Computes dot_hat for a real node using the same coeff codec + scaling semantics
// as stlq runtime scanning (see recall_linkage_disk.cpp).
float DotHatRealByPathCoeffCodec(const stlq::eval::ClusterView& cv,
                                 int cid,
                                 int local_pos,
                                 const std::vector<int>& offsets_root_small,
                                 const std::vector<int>& offsets_one,
                                 const QueryTablesView& qv) {
    if (local_pos < 0 || local_pos >= cv.n_real) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    if (!cv.q_layer_major || !cv.scales_root || !cv.scales_linkage) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    if (!cv.codes_small_bytes || !cv.code0_one_bytes) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    if (cv.n_virt > 0 && !cv.virt_codes_small_bytes) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    if (!qv.xCq_root0_col || !qv.xCq_root_small_col || !qv.xCq_one_col) {
        return std::numeric_limits<float>::quiet_NaN();
    }

    const int m = cv.m;
    const int m_codes = cv.m_codes;
    const int n_virt = cv.n_virt;
    const int nc = cv.nc;
    const int n_root_real = cv.n_root_real;
    if (m <= 1 || m_codes != std::max(0, m - 1) || nc != cv.n_real + cv.n_virt) {
        return std::numeric_limits<float>::quiet_NaN();
    }

    const int stride_codes = m_codes;

    // Walk parent pointers up to a root/virtual node, then accumulate forward.
    int local = n_virt + local_pos;
    if (local < 0 || local >= nc) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    std::vector<int> stack;
    stack.reserve(32);
    while (true) {
        if (local < n_virt) {
            break;  // virtual root
        }
        const int pos = local - n_virt;
        if (pos < 0 || pos >= cv.n_real) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        if (pos < n_root_real) {
            break;  // real root
        }
        stack.push_back(local);
        const std::uint32_t p1 = ReadParent1BasedAt(cv, pos);
        if (p1 == 0u) {
            // Unexpected for a linkage node, but best-effort: treat as root.
            break;
        }
        local = static_cast<int>(p1) - 1;
        if (local < 0 || local >= nc) {
            return std::numeric_limits<float>::quiet_NaN();
        }
    }

    auto q_at = [&](int layer, int local_idx) -> float {
        return static_cast<float>(cv.q_layer_major[static_cast<std::size_t>(layer) * static_cast<std::size_t>(nc) +
                                                   static_cast<std::size_t>(local_idx)]);
    };

    // Base dot for a root/virtual node.
    float dot = 0.0f;
    {
        const float root0 = qv.xCq_root0_col[cid];
        dot += (cv.scales_root[0] * q_at(0, local)) * root0;

        for (int l = 1; l < m; ++l) {
            const int off = offsets_root_small[l];
            const std::uint8_t code = (local < n_virt)
                ? cv.virt_codes_small_bytes[static_cast<std::size_t>(local) * static_cast<std::size_t>(stride_codes) +
                                            static_cast<std::size_t>(l - 1)]
                : cv.codes_small_bytes[static_cast<std::size_t>(local - n_virt) * static_cast<std::size_t>(stride_codes) +
                                       static_cast<std::size_t>(l - 1)];
            const float t = qv.xCq_root_small_col[off + static_cast<int>(code)];
            dot += q_at(l, local) * (cv.scales_root[l] * t);
        }
    }

    // Propagate back down the linkage nodes on the path.
    while (!stack.empty()) {
        const int child_local = stack.back();
        stack.pop_back();
        const int pos = child_local - n_virt;
        if (pos < 0 || pos >= cv.n_real) {
            return std::numeric_limits<float>::quiet_NaN();
        }

        float own = 0.0f;
        const std::uint8_t code0 = cv.code0_one_bytes[pos];
        own += q_at(0, child_local) * (cv.scales_linkage[0] * qv.xCq_one_col[offsets_one[0] + static_cast<int>(code0)]);
        const std::uint8_t* row = cv.codes_small_bytes + static_cast<std::size_t>(pos) * static_cast<std::size_t>(stride_codes);
        for (int l = 1; l < m; ++l) {
            const std::uint8_t code = row[l - 1];
            own += q_at(l, child_local) * (cv.scales_linkage[l] * qv.xCq_one_col[offsets_one[l] + static_cast<int>(code)]);
        }
        dot += own;
    }

    return dot;
}

// Reconstruct x_hat (in the same rotated space as codebooks) for a real node.
// This uses the int8 coeff codec arrays (q_layer_major + scales_root/scales_linkage) and the
// same root/linkage semantics as recall_linkage_disk.cpp.
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
    if (cid < 0 || cid >= C_root0.cols) return false;
    if (C_root0.rows <= 0) return false;
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

    // Build path stack: root (virtual or real root) -> ... -> target linkage node.
    int local = n_virt + local_pos;
    if (local < 0 || local >= nc) return false;
    std::vector<int> linkage_stack;
    linkage_stack.reserve(32);
    while (true) {
        if (local < n_virt) break;  // virtual root
        const int pos = local - n_virt;
        if (pos < 0 || pos >= cv.n_real) return false;
        if (pos < cv.n_root_real) break;  // real root
        linkage_stack.push_back(local);
        const std::uint32_t p1 = ReadParent1BasedAt(cv, pos);
        if (p1 == 0u) break;
        local = static_cast<int>(p1) - 1;
        if (local < 0 || local >= nc) return false;
    }
    const int root_local = local;

    out_xhat->assign(static_cast<std::size_t>(d), 0.0f);
    float* __restrict xhat = out_xhat->data();

    // Root contribution: scales_root * q(layer, root_local) * codebook_vector.
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

    // Linkage residuals: add each node's one-codebook contribution along root->target path.
    while (!linkage_stack.empty()) {
        const int child_local = linkage_stack.back();
        linkage_stack.pop_back();
        const int pos = child_local - n_virt;
        if (pos < 0 || pos >= cv.n_real) return false;

        const std::uint8_t code0 = cv.code0_one_bytes[pos];
        const float a0 = cv.scales_linkage[0] * q_i8(0, child_local);
        AddScaledVectorF32(a0, meta_one.flat.Col(meta_one.offsets[0] + static_cast<int>(code0)), d, xhat);

        const std::uint8_t* __restrict row =
            cv.codes_small_bytes + static_cast<std::size_t>(pos) * static_cast<std::size_t>(stride_codes);
        for (int l = 1; l < m; ++l) {
            const std::uint8_t code = row[l - 1];
            const float a = cv.scales_linkage[l] * q_i8(l, child_local);
            AddScaledVectorF32(a, meta_one.flat.Col(meta_one.offsets[static_cast<std::size_t>(l)] + static_cast<int>(code)), d, xhat);
        }
    }

    return true;
}

// Reconstruct x_hat (in the same rotated space as codebooks) for a real node.
// Uses float coeff arrays from linkage_list (a0.f32 + coeffs.f32) + codes and accumulates along the
// parent linkage. This matches the Julia "len_rate_each" linkaged reconstruction semantics.
[[maybe_unused]] bool ReconstructRealNodeFloatByPath(const STLQGlobalArrays& A,
                                    const ClusterCache& cc,
                                    int cid,
                                    int local_pos,
                                    const stlq::ColMajorMatrix<float>& C_root0,
                                    const stlq::CodebookMeta& meta_root_small,
                                    const stlq::CodebookMeta& meta_one,
                                    std::vector<float>* out_xhat) {
    if (!out_xhat) return false;
    out_xhat->clear();
    if (!A.has_float_coeffs) return false;
    if (local_pos < 0 || local_pos >= cc.n_real) return false;
    if (A.m <= 0 || A.m_codes != std::max(0, A.m - 1)) return false;
    if (C_root0.rows <= 0 || C_root0.cols <= 0) return false;
    if (cid < 0 || cid >= C_root0.cols) return false;
    if (meta_root_small.flat.rows != C_root0.rows || meta_one.flat.rows != C_root0.rows) return false;

    const int d = C_root0.rows;
    out_xhat->assign(static_cast<std::size_t>(d), 0.0f);
    float* __restrict xhat = out_xhat->data();

    const int nc = cc.n_real + cc.n_virt;
    int local = cc.n_virt + local_pos;
    while (true) {
        if (local < 0 || local >= nc) return false;

        if (local < cc.n_virt) {
            // Virtual root contribution (root codebooks).
            const std::size_t gv = static_cast<std::size_t>(cc.virt_lo) + static_cast<std::size_t>(local);
            if (gv >= A.virt_a0.size()) return false;
            const float a0 = A.virt_a0[gv];
            AddScaledVectorF32(a0, C_root0.Col(cid), d, xhat);

            const std::size_t base_codes = gv * static_cast<std::size_t>(A.m_codes);
            const std::size_t base_coeffs = base_codes;
            if (base_codes + static_cast<std::size_t>(A.m_codes) > A.virt_codes_small.size()) return false;
            if (base_coeffs + static_cast<std::size_t>(A.m_codes) > A.virt_coeffs_small.size()) return false;
            for (int l = 1; l < A.m; ++l) {
                const int off = meta_root_small.offsets[static_cast<std::size_t>(l - 1)];
                const std::uint8_t code = A.virt_codes_small[base_codes + static_cast<std::size_t>(l - 1)];
                const float coeff = A.virt_coeffs_small[base_coeffs + static_cast<std::size_t>(l - 1)];
                AddScaledVectorF32(coeff, meta_root_small.flat.Col(off + static_cast<int>(code)), d, xhat);
            }
            break;
        }

        const int pos = local - cc.n_virt;
        if (pos < 0 || pos >= cc.n_real) return false;
        const std::size_t gr = static_cast<std::size_t>(cc.real_lo) + static_cast<std::size_t>(pos);
        if (gr >= A.a0.size()) return false;

        if (pos < cc.n_root_real) {
            // Real root contribution (root codebooks).
            const float a0 = A.a0[gr];
            AddScaledVectorF32(a0, C_root0.Col(cid), d, xhat);

            const std::size_t base_codes = gr * static_cast<std::size_t>(A.m_codes);
            const std::size_t base_coeffs = base_codes;
            if (base_codes + static_cast<std::size_t>(A.m_codes) > A.codes_small.size()) return false;
            if (base_coeffs + static_cast<std::size_t>(A.m_codes) > A.coeffs_small.size()) return false;
            for (int l = 1; l < A.m; ++l) {
                const int off = meta_root_small.offsets[static_cast<std::size_t>(l - 1)];
                const std::uint8_t code = A.codes_small[base_codes + static_cast<std::size_t>(l - 1)];
                const float coeff = A.coeffs_small[base_coeffs + static_cast<std::size_t>(l - 1)];
                AddScaledVectorF32(coeff, meta_root_small.flat.Col(off + static_cast<int>(code)), d, xhat);
            }
            break;
        }

        // Linkage residual contribution for this node (one codebooks).
        {
            const int code0 = ReadCode0One(A.code0_one.data(), static_cast<int>(gr), A.code0_width_bytes);
            const float a0 = A.a0[gr];
            AddScaledVectorF32(a0, meta_one.flat.Col(meta_one.offsets[0] + code0), d, xhat);

            const std::size_t base_codes = gr * static_cast<std::size_t>(A.m_codes);
            const std::size_t base_coeffs = base_codes;
            if (base_codes + static_cast<std::size_t>(A.m_codes) > A.codes_small.size()) return false;
            if (base_coeffs + static_cast<std::size_t>(A.m_codes) > A.coeffs_small.size()) return false;
            for (int l = 1; l < A.m; ++l) {
                const std::uint8_t code = A.codes_small[base_codes + static_cast<std::size_t>(l - 1)];
                const float coeff = A.coeffs_small[base_coeffs + static_cast<std::size_t>(l - 1)];
                AddScaledVectorF32(coeff,
                                   meta_one.flat.Col(meta_one.offsets[static_cast<std::size_t>(l)] + static_cast<int>(code)),
                                   d,
                                   xhat);
            }
        }

        const std::uint32_t p1 = (pos >= 0 && pos < static_cast<int>(cc.parent_real_1based.size()))
                                     ? cc.parent_real_1based[static_cast<std::size_t>(pos)]
                                     : 0u;
        if (p1 == 0u) break;
        local = static_cast<int>(p1) - 1;
    }

    return true;
}

float ResidualVirtRoot(const STLQGlobalArrays& A,
                       const ClusterCache& cc,
                       int local_virt,
                       const std::vector<int>& offsets_root_small,
                       const QueryTablesView& qv,
                       int cid) {
    const std::size_t gv = static_cast<std::size_t>(cc.virt_lo) + static_cast<std::size_t>(local_virt);
    float dot = 0.0f;
    const float root0 = qv.xCq_root0_col[cid];
    dot += A.virt_a0[gv] * root0;
    const std::size_t base_codes = gv * static_cast<std::size_t>(A.m_codes);
    const std::size_t base_coeffs = base_codes;
    for (int l = 1; l < A.m; ++l) {
        const int off = offsets_root_small[l];
        const float* table = qv.xCq_root_small_col + off;
        const std::uint8_t code = A.virt_codes_small[base_codes + static_cast<std::size_t>(l - 1)];
        const float coeff = A.virt_coeffs_small[base_coeffs + static_cast<std::size_t>(l - 1)];
        dot += coeff * table[static_cast<int>(code)];
    }
    return dot;
}

float ResidualRealRoot(const STLQGlobalArrays& A,
                       const ClusterCache& cc,
                       int local_pos,
                       const std::vector<int>& offsets_root_small,
                       const QueryTablesView& qv,
                       int cid) {
    const std::size_t gr = static_cast<std::size_t>(cc.real_lo) + static_cast<std::size_t>(local_pos);
    float dot = 0.0f;
    const float root0 = qv.xCq_root0_col[cid];
    dot += A.a0[gr] * root0;
    const std::size_t base_codes = gr * static_cast<std::size_t>(A.m_codes);
    const std::size_t base_coeffs = base_codes;
    for (int l = 1; l < A.m; ++l) {
        const int off = offsets_root_small[l];
        const float* table = qv.xCq_root_small_col + off;
        const std::uint8_t code = A.codes_small[base_codes + static_cast<std::size_t>(l - 1)];
        const float coeff = A.coeffs_small[base_coeffs + static_cast<std::size_t>(l - 1)];
        dot += coeff * table[static_cast<int>(code)];
    }
    return dot;
}

float ResidualRealLinkage(const STLQGlobalArrays& A,
                        const ClusterCache& cc,
                        int local_pos,
                        const stlq::CodebookMeta& meta_one,
                        const QueryTablesView& qv) {
    const std::size_t gr = static_cast<std::size_t>(cc.real_lo) + static_cast<std::size_t>(local_pos);
    float dot = 0.0f;
    const int code0 = ReadCode0One(A.code0_one.data(), static_cast<int>(gr), A.code0_width_bytes);
    const float* one0 = qv.xCq_one_col + meta_one.offsets[0];
    dot += A.a0[gr] * one0[code0];

    const std::size_t base_codes = gr * static_cast<std::size_t>(A.m_codes);
    const std::size_t base_coeffs = base_codes;
    for (int l = 1; l < A.m; ++l) {
        const float* one_l = qv.xCq_one_col + meta_one.offsets[static_cast<std::size_t>(l)];
        const std::uint8_t code = A.codes_small[base_codes + static_cast<std::size_t>(l - 1)];
        const float coeff = A.coeffs_small[base_coeffs + static_cast<std::size_t>(l - 1)];
        dot += coeff * one_l[static_cast<int>(code)];
    }
    return dot;
}

[[maybe_unused]] float DotHatRealByPath(const STLQGlobalArrays& A,
                       const ClusterCache& cc,
                       int cid,
                       int local_pos,
                       const std::vector<int>& offsets_root_small,
                       const stlq::CodebookMeta& meta_one,
                       const QueryTablesView& qv) {
    float sum = 0.0f;
    const int nc = cc.n_real + cc.n_virt;
    int local = cc.n_virt + local_pos;
    while (true) {
        if (local < 0 || local >= nc) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        if (local < cc.n_virt) {
            sum += ResidualVirtRoot(A, cc, local, offsets_root_small, qv, cid);
            break;
        }
        const int pos = local - cc.n_virt;
        if (pos < cc.n_root_real) {
            sum += ResidualRealRoot(A, cc, pos, offsets_root_small, qv, cid);
            break;
        }
        sum += ResidualRealLinkage(A, cc, pos, meta_one, qv);
        const std::uint32_t p1 = (pos >= 0 && pos < static_cast<int>(cc.parent_real_1based.size()))
                                     ? cc.parent_real_1based[static_cast<std::size_t>(pos)]
                                     : 0u;
        if (p1 == 0u) {
            break;
        }
        local = static_cast<int>(p1) - 1;
    }
    return sum;
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
    LogInfo("Loading config from: " + cfg_path.string());

    Config cfg = DefaultConfig(true);
    NormalizeHVec(&cfg);
    std::vector<int> ignored;
    if (!LoadConfigFile(cfg_path.string(), &cfg, &err, &ignored)) {
        LogError(err);
        return 1;
    }

    // Load train model (codebooks).
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
            LogInfo("Exporter using run_state effective_train_h5: " + effective_train_h5);
        }
        base_path = ResolveLoadBasePath(base_path, load_date, load_seq, run_root, cfg_path);
        if (!io::LoadTrainResults(base_path, load_date, load_seq, cfg.model.m, &train, &train_err)) {
            LogError(train_err.empty() ? "LoadTrainResults failed." : train_err);
            return 1;
        }
    }

    // Load queries (in-mem) to build query tables.
    ColMajorMatrix<float> Xq;
    std::vector<int> gt_first_ignored;
    if (!io::LoadQuerySet(cfg, &Xq, &gt_first_ignored, &err)) {
        LogError(err);
        return 1;
    }

    // IMPORTANT: stlq codebooks / linkage_list codes are in the rotated (OPQ) space.
    // For exporter correctness, build query tables from rotated queries as well.
    if (!IsIdentityRotation(train.R)) {
        LogInfo("Rotating queries by train.R (OPQ) for exporter alignment...");
        ApplyRotationInPlace(train.R, &Xq);
    }

    io::DatasetGroundtruthReader gt_reader;
    if (!io::OpenDatasetGroundtruthReader(cfg, &gt_reader, &err)) {
        LogError(err);
        return 1;
    }
    const int gt_k = gt_reader.k;
    const int topL = std::min(args.gt_topL, std::max(1, gt_k));

    const int nq = Xq.cols;
    ColMajorMatrix<int> gt_mat;
    if (!io::ReadDatasetGroundtruthTopK(cfg, gt_reader, nq, topL, &gt_mat, &err)) {
        LogError(err);
        return 1;
    }
    if (gt_mat.cols != nq) {
        LogError("Groundtruth rows mismatch with query count.");
        return 1;
    }

    std::vector<std::uint32_t> gt_ids_flat(static_cast<std::size_t>(nq) * static_cast<std::size_t>(topL));
    for (int q = 0; q < nq; ++q) {
        const int* col = gt_mat.Col(q);
        for (int i = 0; i < topL; ++i) {
            int v = col[i];
            gt_ids_flat[static_cast<std::size_t>(q) * static_cast<std::size_t>(topL) + static_cast<std::size_t>(i)] =
                static_cast<std::uint32_t>(std::max(0, v));
        }
    }

    // Open linkage_list.
    io::LinkageListReader linkage_list;
    const std::filesystem::path linkage_list_dir = run_root / "linkage_list";
    if (!linkage_list.Open(linkage_list_dir.string(), &err)) {
        LogError(err);
        return 1;
    }

    // Optional: open coeff codec and use ClusterProvider to match runtime scan semantics.
    bool use_coeff_codec = false;
    io::LinkageCoeffCodecReader coeff_reader;
    eval::ClusterProvider cluster_provider;
    {
        const std::filesystem::path coeff_meta = linkage_list_dir / "coeff_meta.bin";
        if (std::filesystem::exists(coeff_meta)) {
            std::string local_err;
            if (coeff_reader.Open(linkage_list_dir.string(), &local_err)) {
                use_coeff_codec = true;
                // We still need ClusterProvider to materialize ClusterView (real_ids, codes, q_layer_major, scales,
                // parent). Norm2 can be stored either as a direct float file (norm2_int8.f32) or as a LUT
                // (norm2_int8_lut_codes.u8 + norm2_int8_lut_centers.f32). Auto-detect what exists to avoid
                // per-cluster open failures and warning spam.

                const std::filesystem::path norm2_direct_f32 = linkage_list_dir / "norm2_int8.f32";
                const std::filesystem::path norm2_lut_codes_u8 = linkage_list_dir / "norm2_int8_lut_codes.u8";
                const std::filesystem::path norm2_lut_centers_f32 = linkage_list_dir / "norm2_int8_lut_centers.f32";
                const bool has_norm2_direct_f32 = std::filesystem::exists(norm2_direct_f32);
                const bool has_norm2_lut = std::filesystem::exists(norm2_lut_codes_u8) && std::filesystem::exists(norm2_lut_centers_f32);
                const bool use_norm2_lut = (!has_norm2_direct_f32) && has_norm2_lut;
                const bool use_norm2_lut_global = use_norm2_lut;
                std::uint32_t norm2_lut_h = 256;
                if (use_norm2_lut) {
                    std::error_code fec;
                    const std::uintmax_t fsz = std::filesystem::file_size(norm2_lut_centers_f32, fec);
                    if (!fec && fsz > 0 && (fsz % sizeof(float) == 0)) {
                        const std::uintmax_t cnt = fsz / sizeof(float);
                        if (cnt > 0 && cnt <= 4096) {
                            norm2_lut_h = static_cast<std::uint32_t>(cnt);
                        }
                    }
                }

                if (!cluster_provider.Open(linkage_list,
                                           &coeff_reader,
                                           /*norm_provider=*/nullptr,
                                           /*use_coeff_codec=*/true,
                                           /*prefer_parent_louds=*/true,
                                           static_cast<std::uint32_t>(cfg.eval.parent_louds_select_stride),
                                           static_cast<std::uint32_t>(cfg.eval.parent_louds_rank_words_per_super_log2),
                                           /*parent_louds_build_indices=*/cfg.eval.parent_louds_build_indices,
                                           /*adaptive_parent_u16_storage=*/false,
                                           /*use_norm2_lut=*/use_norm2_lut,
                                           /*use_norm2_lut_global=*/use_norm2_lut_global,
                                           /*norm2_lut_h=*/norm2_lut_h,
                                           /*norm2_lut_kmeans_niter=*/25,
                                           /*profile_prep_stats=*/false,
                                           &local_err)) {
                    LogError("Coeff codec present but ClusterProvider.Open failed: " + local_err);
                    return 1;
                } else {
                    // NormProvider is null. If we want ClusterProvider to work, we must allow disk-lazy
                    // norm2 loading whenever we rely on on-disk norm2 artifacts (direct f32 or LUT).
                    // For global LUT, we must preload centers before GetCluster.
                    if (use_norm2_lut_global) {
                        std::vector<float> centers;
                        std::string centers_err;
                        if (ReadBinaryFileExact<float>(norm2_lut_centers_f32,
                                                       static_cast<std::size_t>(norm2_lut_h),
                                                       &centers,
                                                       &centers_err)) {
                            cluster_provider.SetGlobalNorm2LutCenters(std::move(centers));
                        } else {
                            LogWarn("Failed to preload global norm2 LUT centers: " + centers_err);
                        }
                    }
                    cluster_provider.SetNorm2DiskLazyEnabled(has_norm2_direct_f32 || use_norm2_lut);
                    LogInfo("Exporter using coeff codec (q_layer_major + scales) for dot_hat (runtime-aligned).");
                }
            } else {
                LogError("Coeff codec meta exists but failed to open: " + local_err);
                return 1;
            }
        }
    }

    if (!use_coeff_codec) {
        LogError("This exporter run is configured to require coeff codec (int8 coefficients decoded from Huffman files), but coeff_meta.bin was not found.");
        return 1;
    }

    // Global arrays.
    STLQGlobalArrays A;
    if (!LoadSTLQGlobalArrays(linkage_list_dir, linkage_list, cfg.model.m, use_coeff_codec, &A, &err)) {
        LogError(err);
        return 1;
    }

    // Build gid -> global real index (depth-order index in [0,total_real)).
    std::vector<std::int32_t> gid_to_real_index(static_cast<std::size_t>(cfg.dataset.nbase), -1);
    for (std::uint64_t i = 0; i < A.total_real; ++i) {
        const std::uint32_t gid = A.real_ids[static_cast<std::size_t>(i)];
        if (gid < gid_to_real_index.size()) {
            gid_to_real_index[static_cast<std::size_t>(gid)] = static_cast<std::int32_t>(i);
        }
    }

    // Build real_index -> cid (for fast lookup).
    std::vector<int> real_index_to_cid(static_cast<std::size_t>(A.total_real), -1);
    const auto& real_offsets = linkage_list.real_offsets();
    for (int cid = 0; cid < linkage_list.nlist(); ++cid) {
        const std::uint64_t lo = real_offsets[static_cast<std::size_t>(cid)];
        const std::uint64_t hi = real_offsets[static_cast<std::size_t>(cid + 1)];
        for (std::uint64_t idx = lo; idx < hi; ++idx) {
            real_index_to_cid[static_cast<std::size_t>(idx)] = cid;
        }
    }

    // Build codebook meta for query tables.
    if (train.C_root.books.empty() || train.C_one.books.empty()) {
        LogError("TrainResult missing codebooks (C_root/C_one). Exporter requires full linkage model.");
        return 1;
    }
    const ColMajorMatrix<float>& C_root0 = train.C_root.books.front();
    std::vector<const ColMajorMatrix<float>*> root_small_books;
    root_small_books.reserve(static_cast<std::size_t>(A.m_codes));
    for (int l = 1; l < cfg.model.m; ++l) {
        root_small_books.push_back(&train.C_root.books[static_cast<std::size_t>(l)]);
    }
    const CodebookMeta meta_root_small = BuildCodebookMeta(root_small_books);
    const CodebookMeta meta_one = BuildCodebookMeta(GatherBooks(train.C_one));

    std::vector<int> offsets_root_small(static_cast<std::size_t>(cfg.model.m), 0);
    for (int l = 1; l < cfg.model.m; ++l) {
        offsets_root_small[static_cast<std::size_t>(l)] = meta_root_small.offsets[static_cast<std::size_t>(l - 1)];
    }

    // Prepare output.
    const std::filesystem::path out_dir(args.out_dir);
    std::error_code ec;
    std::filesystem::create_directories(out_dir, ec);
    if (ec) {
        LogError("Failed to create out_dir: " + out_dir.string());
        return 1;
    }

    std::vector<float> dot_hat_flat(gt_ids_flat.size(), std::numeric_limits<float>::quiet_NaN());
    // Dot of the explicit linkaged reconstruction implied by the coeff codec (int8+scale, Huffman-decoded).
    std::vector<float> dot_recon_flat(gt_ids_flat.size(), std::numeric_limits<float>::quiet_NaN());
    std::vector<float> norm_hat_flat(gt_ids_flat.size(), std::numeric_limits<float>::quiet_NaN());
    std::vector<float> mse_recon_raw_flat(gt_ids_flat.size(), std::numeric_limits<float>::quiet_NaN());
    std::vector<float> raw_norm2_flat(gt_ids_flat.size(), std::numeric_limits<float>::quiet_NaN());
    std::vector<float> raw_dot_recon_flat(gt_ids_flat.size(), std::numeric_limits<float>::quiet_NaN());
    std::vector<float> raw_dot_query_flat(gt_ids_flat.size(), std::numeric_limits<float>::quiet_NaN());

    // Norm2 of the final reconstructed vector implied by the linkage (same space as codebooks).
    // Note: This is NOT the same as stlq's r_norm2 term used at runtime.
    std::vector<float> norm2_recon_flat(gt_ids_flat.size(), std::numeric_limits<float>::quiet_NaN());

    // Optional: export the exact runtime score form (omit ||q||^2 constant): d_hat_noq = ||x_hat||^2 - 2*dot_hat.
    std::vector<float> d_hat_noq_flat(gt_ids_flat.size(), std::numeric_limits<float>::quiet_NaN());

    std::atomic<std::uint64_t> bad_pos_mismatches{0};
    std::atomic<std::uint64_t> getcluster_failures{0};
    const int d = Xq.rows;

    stlq::io::DatasetVectorReader base_r;
    bool have_base_reader = false;
    {
        std::string open_err;
        if (!stlq::io::OpenDatasetVectorReader(cfg, stlq::io::DatasetRole::kBase, &base_r, &open_err)) {
            LogWarn("Failed to open base reader for recon-vs-raw MSE export: " + open_err);
        } else if (base_r.d != d) {
            LogWarn("Base dim mismatch for recon-vs-raw MSE export (base_d=" +
                    std::to_string(base_r.d) + ", q_d=" + std::to_string(d) + ")");
        } else {
            have_base_reader = true;
        }
    }

    const int qblk = args.q_block;
    const int blas_threads = std::max(1, OmpMaxThreads());

    Timer t_total;
    for (int q0 = 0; q0 < nq; q0 += qblk) {
        const int qlen = std::min(qblk, nq - q0);
        const float* Xq_blk = Xq.Col(q0);

        double t_coarse = 0.0, t_root_small = 0.0, t_one = 0.0;
        STLQueryTables qt = BuildSTLQueryTables(
            Xq_blk,
            /*ldXq=*/d,
            d,
            qlen,
            C_root0,
            meta_root_small,
            meta_one,
            blas_threads,
            &t_coarse,
            &t_root_small,
            &t_one);

        const int h0 = qt.xCq_root0.rows;
        const int hrs = qt.xCq_root_small.rows;
        const int ho = qt.xCq_one.rows;
        (void)hrs;
        (void)ho;

        for (int qi = 0; qi < qlen; ++qi) {
            const int q = q0 + qi;
            const float* qvec = Xq_blk + static_cast<std::ptrdiff_t>(qi) * static_cast<std::ptrdiff_t>(d);
            QueryTablesView qv;
            qv.xCq_root0_col = qt.xCq_root0.data.data() + static_cast<std::size_t>(qi) * static_cast<std::size_t>(h0);
            qv.xCq_root_small_col = qt.xCq_root_small.data.data() +
                                    static_cast<std::size_t>(qi) * static_cast<std::size_t>(qt.xCq_root_small.rows);
            qv.xCq_one_col = qt.xCq_one.data.data() + static_cast<std::size_t>(qi) * static_cast<std::size_t>(qt.xCq_one.rows);

            for (int i = 0; i < topL; ++i) {
                const std::size_t out_idx =
                    static_cast<std::size_t>(q) * static_cast<std::size_t>(topL) + static_cast<std::size_t>(i);
                const std::uint32_t gid = gt_ids_flat[out_idx];
                if (gid >= gid_to_real_index.size()) {
                    continue;
                }
                const std::int32_t real_idx = gid_to_real_index[static_cast<std::size_t>(gid)];
                if (real_idx < 0) {
                    continue;
                }
                const int cid = real_index_to_cid[static_cast<std::size_t>(real_idx)];
                if (cid < 0) {
                    continue;
                }
                float dot_hat = std::numeric_limits<float>::quiet_NaN();
                float dot_recon = std::numeric_limits<float>::quiet_NaN();
                {
                    stlq::eval::ClusterView cv;
                    stlq::eval::ClusterProvider::PrepStats ps;
                    std::string local_err;
                    if (cluster_provider.GetCluster(cid, &cv, &ps, &local_err)) {
                        int local_pos = static_cast<int>(static_cast<std::uint64_t>(real_idx) -
                                                         linkage_list.real_offsets()[static_cast<std::size_t>(cid)]);
                        // Validate that (cid,local_pos) actually corresponds to this gid.
                        // If not, locate the correct position by scanning cv.real_ids.
                        if (!cv.real_ids || local_pos < 0 || local_pos >= cv.n_real ||
                            cv.real_ids[local_pos] != gid) {
                            int found = -1;
                            if (cv.real_ids) {
                                for (int p = 0; p < cv.n_real; ++p) {
                                    if (cv.real_ids[p] == gid) {
                                        found = p;
                                        break;
                                    }
                                }
                            }
                            bad_pos_mismatches.fetch_add(1, std::memory_order_relaxed);
                            if (found >= 0) {
                                local_pos = found;
                            } else {
                                // Can't map gid into this cluster view; skip.
                                continue;
                            }
                        }

                        dot_hat = DotHatRealByPathCoeffCodec(cv,
                                                             cid,
                                                             local_pos,
                                                             offsets_root_small,
                                                             meta_one.offsets,
                                                             qv);

                        // Explicit reconstruction in codebook (rotated) space from Huffman-decoded int8 coeffs.
                        std::vector<float> xhat;
                        if (ReconstructRealNodeCoeffCodec(cv,
                                                          cid,
                                                          local_pos,
                                                          C_root0,
                                                          meta_root_small,
                                                          meta_one,
                                                          &xhat)) {
                            dot_recon = DotF32(qvec, xhat.data(), d);
                            norm2_recon_flat[out_idx] = Norm2F32(xhat.data(), d);
                            if (have_base_reader &&
                                gid < static_cast<std::uint32_t>(cfg.dataset.nbase)) {
                                stlq::ColMajorMatrix<float> xraw;
                                std::string read_one_err;
                                if (stlq::io::ReadDatasetVectorByIdsF32(base_r, std::vector<std::uint32_t>{gid}, &xraw, &read_one_err) &&
                                    xraw.cols == 1 && xraw.rows == d) {
                                    if (!stlq::IsIdentityRotation(train.R)) {
                                        stlq::ApplyRotationInPlace(train.R, &xraw);
                                    }
                                    raw_norm2_flat[out_idx] = Norm2F32(xraw.Col(0), d);
                                    raw_dot_recon_flat[out_idx] = DotF32(xraw.Col(0), xhat.data(), d);
                                    raw_dot_query_flat[out_idx] = DotF32(qvec, xraw.Col(0), d);
                                    double mse = 0.0;
                                    for (int r = 0; r < d; ++r) {
                                        const double diff = static_cast<double>(xhat[static_cast<std::size_t>(r)]) -
                                                            static_cast<double>(xraw(r, 0));
                                        mse += diff * diff;
                                    }
                                    mse_recon_raw_flat[out_idx] =
                                        static_cast<float>(mse / static_cast<double>(std::max(1, d)));
                                }
                            }
                        }

                        // Prefer norm2 from the same ClusterView (keeps dot/norm aligned to the same node).
                        if (cv.r_norm2) {
                            norm_hat_flat[out_idx] = cv.r_norm2[local_pos];
                        } else if (cv.r_norm2_lut_u8 && cv.r_norm2_lut_centers && cv.r_norm2_lut_size > 0) {
                            const std::uint8_t c = cv.r_norm2_lut_u8[local_pos];
                            if (static_cast<int>(c) < cv.r_norm2_lut_size) {
                                norm_hat_flat[out_idx] = cv.r_norm2_lut_centers[static_cast<int>(c)];
                            }
                        }
                    } else {
                        const std::uint64_t nfail = getcluster_failures.fetch_add(1, std::memory_order_relaxed) + 1;
                        if (nfail <= 5) {
                            LogWarn("ClusterProvider.GetCluster failed for cid=" + std::to_string(cid) + ": " + local_err);
                        }
                    }
                }
                if (!std::isfinite(dot_hat)) {
                    // Strict mode: do not fall back to float coeffs.
                    continue;
                }
                dot_hat_flat[out_idx] = dot_hat;
                dot_recon_flat[out_idx] = dot_recon;
                // If coeff codec path already filled norm_hat from ClusterView, keep it.
                if (!std::isfinite(norm_hat_flat[out_idx])) {
                    if (A.has_norm2_direct && static_cast<std::size_t>(real_idx) < A.norm2_direct_f32.size()) {
                        norm_hat_flat[out_idx] = A.norm2_direct_f32[static_cast<std::size_t>(real_idx)];
                    } else if (!A.norm2_codes_u8.empty() && !A.norm2_centers_f32.empty() &&
                               static_cast<std::size_t>(real_idx) < A.norm2_codes_u8.size()) {
                        const std::uint8_t nc = A.norm2_codes_u8[static_cast<std::size_t>(real_idx)];
                        if (static_cast<std::size_t>(nc) < A.norm2_centers_f32.size()) {
                            norm_hat_flat[out_idx] = A.norm2_centers_f32[static_cast<std::size_t>(nc)];
                        }
                    }
                }

                if (std::isfinite(dot_hat_flat[out_idx]) && std::isfinite(norm_hat_flat[out_idx])) {
                    d_hat_noq_flat[out_idx] = norm_hat_flat[out_idx] - 2.0f * dot_hat_flat[out_idx];
                }
            }
        }
        LogInfo("Processed queries [" + std::to_string(q0) + "," + std::to_string(q0 + qlen) + ")");
    }

    const std::uint64_t mism = bad_pos_mismatches.load(std::memory_order_relaxed);
    if (mism > 0) {
        LogWarn("Exporter detected gid->local_pos mismatches (fixed by scanning ClusterView.real_ids): " + std::to_string(mism));
    }
    const std::uint64_t nfail = getcluster_failures.load(std::memory_order_relaxed);
    if (nfail > 0) {
        LogWarn("ClusterProvider.GetCluster failures: " + std::to_string(nfail) + " (only first 5 shown)");
    }

    const double sec = t_total.ElapsedSeconds();
    LogInfo("Export done in " + std::to_string(sec) + "s");

    // ---- Sanity check: reconstruct x_hat using the eval semantics and compare multiple "truths" ----
    if (have_base_reader) {
            const int q_sanity = 0;
            const int want = std::min(topL, 16);
            std::vector<std::uint32_t> ids;
            ids.reserve(static_cast<std::size_t>(want));
            for (int i = 0; i < want; ++i) {
                const std::size_t idx = static_cast<std::size_t>(q_sanity) * static_cast<std::size_t>(topL) + static_cast<std::size_t>(i);
                const std::uint32_t gid = gt_ids_flat[idx];
                ids.push_back(gid);
            }
            stlq::ColMajorMatrix<float> Xb_samp;
            std::string read_err;
            if (!stlq::io::ReadDatasetVectorByIdsF32(base_r, ids, &Xb_samp, &read_err)) {
                LogWarn("Sanity check: failed reading base ids: " + read_err);
            } else {
                if (!stlq::IsIdentityRotation(train.R)) {
                    stlq::ApplyRotationInPlace(train.R, &Xb_samp);
                }
                const float* qv = Xq.Col(q_sanity);
                double sum_abs_dot_eval_vs_recon = 0.0;
                double sum_abs_norm_eval_vs_recon = 0.0;
                double sum_abs_d_eval_vs_recon = 0.0;
                double sum_abs_dot_eval_vs_raw = 0.0;
                double sum_abs_norm_eval_vs_raw = 0.0;
                double sum_abs_d_eval_vs_raw = 0.0;
                double sum_mse_recon_vs_raw = 0.0;
                double sum_abs_dot_recon_vs_raw = 0.0;
                double sum_abs_norm_recon_vs_raw = 0.0;
                double sum_abs_d_recon_vs_raw = 0.0;
                int got_recon = 0;
                int got_raw = 0;
                for (int j = 0; j < Xb_samp.cols; ++j) {
                    const float* xv = Xb_samp.Col(j);
                    const std::size_t out_idx = static_cast<std::size_t>(q_sanity) * static_cast<std::size_t>(topL) + static_cast<std::size_t>(j);
                    const float dot_hat = dot_hat_flat[out_idx];
                    const float norm_hat = norm_hat_flat[out_idx];
                    if (std::isfinite(dot_hat) && std::isfinite(norm_hat)) {
                        const float dot_raw = DotF32(qv, xv, d);
                        const float norm2_raw = Norm2F32(xv, d);
                        const float d_raw_noq = norm2_raw - 2.0f * dot_raw;
                        const float d_hat_noq = norm_hat - 2.0f * dot_hat;

                        sum_abs_dot_eval_vs_raw += std::abs(static_cast<double>(dot_hat - dot_raw));
                        sum_abs_norm_eval_vs_raw += std::abs(static_cast<double>(norm_hat - norm2_raw));
                        sum_abs_d_eval_vs_raw += std::abs(static_cast<double>(d_hat_noq - d_raw_noq));
                        ++got_raw;

                        if (use_coeff_codec) {
                            const std::uint32_t gid = ids[static_cast<std::size_t>(j)];
                            const std::int32_t real_idx = (gid < gid_to_real_index.size())
                                ? gid_to_real_index[static_cast<std::size_t>(gid)]
                                : -1;
                            if (real_idx >= 0 && static_cast<std::size_t>(real_idx) < real_index_to_cid.size()) {
                                const int cid = real_index_to_cid[static_cast<std::size_t>(real_idx)];
                                if (cid >= 0) {
                                    stlq::eval::ClusterView cv;
                                    stlq::eval::ClusterProvider::PrepStats ps;
                                    std::string local_err;
                                    if (cluster_provider.GetCluster(cid, &cv, &ps, &local_err)) {
                                        int local_pos = static_cast<int>(static_cast<std::uint64_t>(real_idx) -
                                                                         linkage_list.real_offsets()[static_cast<std::size_t>(cid)]);
                                        if (!cv.real_ids || local_pos < 0 || local_pos >= cv.n_real || cv.real_ids[local_pos] != gid) {
                                            int found = -1;
                                            if (cv.real_ids) {
                                                for (int p = 0; p < cv.n_real; ++p) {
                                                    if (cv.real_ids[p] == gid) { found = p; break; }
                                                }
                                            }
                                            if (found >= 0) {
                                                local_pos = found;
                                            } else {
                                                continue;
                                            }
                                        }
                                        std::vector<float> xhat;
                                        if (ReconstructRealNodeCoeffCodec(cv, cid, local_pos, C_root0,
                                                                          meta_root_small, meta_one, &xhat)) {
                                            const float dot_recon = DotF32(qv, xhat.data(), d);
                                            const float norm2_recon = Norm2F32(xhat.data(), d);
                                            const float d_recon_noq = norm2_recon - 2.0f * dot_recon;

                                            sum_abs_dot_eval_vs_recon += std::abs(static_cast<double>(dot_hat - dot_recon));
                                            sum_abs_norm_eval_vs_recon += std::abs(static_cast<double>(norm_hat - norm2_recon));
                                            sum_abs_d_eval_vs_recon += std::abs(static_cast<double>(d_hat_noq - d_recon_noq));

                                            sum_abs_dot_recon_vs_raw += std::abs(static_cast<double>(dot_recon - dot_raw));
                                            sum_abs_norm_recon_vs_raw += std::abs(static_cast<double>(norm2_recon - norm2_raw));
                                            sum_abs_d_recon_vs_raw += std::abs(static_cast<double>(d_recon_noq - d_raw_noq));

                                            double mse = 0.0;
                                            for (int r = 0; r < d; ++r) {
                                                const double diff = static_cast<double>(xhat[static_cast<std::size_t>(r)]) -
                                                                    static_cast<double>(xv[r]);
                                                mse += diff * diff;
                                            }
                                            mse /= static_cast<double>(std::max(1, d));
                                            sum_mse_recon_vs_raw += mse;
                                            ++got_recon;
                        }
                    }
                }
            }
    } else {
        LogWarn("Sanity check: skipped because base reader is unavailable.");
    }
                }
                if (got_raw > 0) {
                    LogInfo("Sanity check (q=0, first " + std::to_string(got_raw) + " GT ids) vs RAW base vectors (rotated): "
                            "mean|dot_hat-dot_raw|=" + std::to_string(sum_abs_dot_eval_vs_raw / got_raw) +
                            "  mean|norm_hat-norm2_raw|=" + std::to_string(sum_abs_norm_eval_vs_raw / got_raw) +
                            "  mean|d_hat_noq-d_raw_noq|=" + std::to_string(sum_abs_d_eval_vs_raw / got_raw));
                }
                if (got_recon > 0) {
                    LogInfo("Sanity check (q=0, first " + std::to_string(got_recon) + " GT ids) vs RECON (coeff-codec semantics): "
                            "mean|dot_hat-dot_recon|=" + std::to_string(sum_abs_dot_eval_vs_recon / got_recon) +
                            "  mean|norm_hat-norm2_recon|=" + std::to_string(sum_abs_norm_eval_vs_recon / got_recon) +
                            "  mean|d_hat_noq-d_recon_noq|=" + std::to_string(sum_abs_d_eval_vs_recon / got_recon) +
                            "  meanMSE(recon,raw)=" + std::to_string(sum_mse_recon_vs_raw / got_recon));

                    LogInfo("Sanity check (q=0, first " + std::to_string(got_recon) + " GT ids) RECON vs RAW (final reconstruction quality): "
                            "mean|dot_recon-dot_raw|=" + std::to_string(sum_abs_dot_recon_vs_raw / got_recon) +
                            "  mean|norm2_recon-norm2_raw|=" + std::to_string(sum_abs_norm_recon_vs_raw / got_recon) +
                            "  mean|d_recon_noq-d_raw_noq|=" + std::to_string(sum_abs_d_recon_vs_raw / got_recon));
                } else if (use_coeff_codec) {
                    LogWarn("Sanity check: recon path did not run (missing coeff codec / cluster views / codebooks?).");
                }
            }
        }
    }

    // Write outputs.
    {
        std::ostringstream meta;
        meta << "format = stlq_gt_pairs_v1\n";
        meta << "run_root = \"" << run_root.string() << "\"\n";
        meta << "linkage_list_dir = \"" << linkage_list_dir.string() << "\"\n";
        meta << "dataset.name = \"" << cfg.dataset.name << "\"\n";
        meta << "dataset.data_root = \"" << cfg.dataset.data_root << "\"\n";
        meta << "dataset.train_path = \"" << cfg.dataset.train_path << "\"\n";
        meta << "dataset.base_path = \"" << cfg.dataset.base_path << "\"\n";
        meta << "dataset.query_path = \"" << cfg.dataset.query_path << "\"\n";
        meta << "dataset.groundtruth_path = \"" << cfg.dataset.groundtruth_path << "\"\n";
        meta << "dataset.groundtruth_add1 = " << cfg.dataset.groundtruth_add1 << "\n";
        meta << "dataset.nbase = " << cfg.dataset.nbase << "\n";
        meta << "dataset.nquery = " << nq << "\n";
        meta << "gt_topL = " << topL << "\n";
        meta << "model.m = " << cfg.model.m << "\n";
        meta << "linkage_list.nlist = " << linkage_list.nlist() << "\n";
        meta << "linkage_list.total_real = " << linkage_list.total_real() << "\n";
        meta << "linkage_list.total_virtual = " << linkage_list.total_virtual() << "\n";
        meta << "eval.parent_louds_select_stride = " << cfg.eval.parent_louds_select_stride << "\n";
        meta << "eval.parent_louds_rank_words_per_super_log2 = " << cfg.eval.parent_louds_rank_words_per_super_log2 << "\n";
        meta << "eval.parent_louds_build_indices = " << (cfg.eval.parent_louds_build_indices ? 1 : 0) << "\n";
        meta << "stlq_export.has_norm2_recon = 1\n";
        meta << "stlq_export.has_dot_recon = 1\n";
        meta << "stlq_export.has_mse_recon_raw = 1\n";
        meta << "stlq_export.has_raw_norm2 = 1\n";
        meta << "stlq_export.has_raw_dot_recon = 1\n";
        meta << "stlq_export.has_raw_dot_query = 1\n";
        meta << "stlq_export.recon_source = \"coeff_codec_huffman\"\n";

        const std::string meta_s = meta.str();
        if (!WriteBinaryFile(out_dir / "meta.txt", meta_s.data(), meta_s.size(), &err)) {
            LogError(err);
            return 1;
        }
    }

    if (!WriteBinaryFile(out_dir / "gt_ids.u32", gt_ids_flat.data(), gt_ids_flat.size() * sizeof(std::uint32_t), &err)) {
        LogError(err);
        return 1;
    }
    if (!WriteBinaryFile(out_dir / "stlq_dot_hat.f32", dot_hat_flat.data(), dot_hat_flat.size() * sizeof(float), &err)) {
        LogError(err);
        return 1;
    }
    if (!WriteBinaryFile(out_dir / "stlq_dot_recon.f32", dot_recon_flat.data(), dot_recon_flat.size() * sizeof(float), &err)) {
        LogError(err);
        return 1;
    }
    if (!WriteBinaryFile(out_dir / "stlq_norm_hat.f32", norm_hat_flat.data(), norm_hat_flat.size() * sizeof(float), &err)) {
        LogError(err);
        return 1;
    }
    if (!WriteBinaryFile(out_dir / "stlq_norm2_recon.f32",
                         norm2_recon_flat.data(),
                         norm2_recon_flat.size() * sizeof(float),
                         &err)) {
        LogError(err);
        return 1;
    }
    if (!WriteBinaryFile(out_dir / "stlq_mse_recon_raw.f32",
                         mse_recon_raw_flat.data(),
                         mse_recon_raw_flat.size() * sizeof(float),
                         &err)) {
        LogError(err);
        return 1;
    }
    if (!WriteBinaryFile(out_dir / "stlq_raw_norm2.f32",
                         raw_norm2_flat.data(),
                         raw_norm2_flat.size() * sizeof(float),
                         &err)) {
        LogError(err);
        return 1;
    }
    if (!WriteBinaryFile(out_dir / "stlq_raw_dot_recon.f32",
                         raw_dot_recon_flat.data(),
                         raw_dot_recon_flat.size() * sizeof(float),
                         &err)) {
        LogError(err);
        return 1;
    }
    if (!WriteBinaryFile(out_dir / "stlq_raw_dot_query.f32",
                         raw_dot_query_flat.data(),
                         raw_dot_query_flat.size() * sizeof(float),
                         &err)) {
        LogError(err);
        return 1;
    }
    if (!WriteBinaryFile(out_dir / "stlq_d_hat_noq.f32", d_hat_noq_flat.data(), d_hat_noq_flat.size() * sizeof(float), &err)) {
        LogError(err);
        return 1;
    }

    LogInfo("Wrote: " + (out_dir / "meta.txt").string());
    LogInfo("Wrote: " + (out_dir / "gt_ids.u32").string());
    LogInfo("Wrote: " + (out_dir / "stlq_dot_hat.f32").string());
    LogInfo("Wrote: " + (out_dir / "stlq_dot_recon.f32").string());
    LogInfo("Wrote: " + (out_dir / "stlq_norm_hat.f32").string());
    LogInfo("Wrote: " + (out_dir / "stlq_mse_recon_raw.f32").string());
    LogInfo("Wrote: " + (out_dir / "stlq_raw_norm2.f32").string());
    LogInfo("Wrote: " + (out_dir / "stlq_raw_dot_recon.f32").string());
    LogInfo("Wrote: " + (out_dir / "stlq_raw_dot_query.f32").string());
    LogInfo("Wrote: " + (out_dir / "stlq_d_hat_noq.f32").string());

    return 0;
}
