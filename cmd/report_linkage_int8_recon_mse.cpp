// Two-phase MSE tool for stlq int8 coeff-codec reconstruction.
//
// Phase 1 (bucketize):
//   - Sequentially scan linkage_list/real_ids.u32 (by cluster) and write temp buckets
//     gid_bucket_<b>.bin containing {gid,cid,pos} records.
//
// Optional fast path (--fast_dp=1):
//   - Phase 1 DP-reconstructs recon vectors per gid and writes temp recon buckets
//     recon_bucket_<b>_t<tid>_f16.bin containing {gid,rnorm2,recon[d]} records (recon stored as fp16).
//
// Phase 2 (sequential scan):
//   - For each gid bucket in increasing order, sequentially read base vectors
//     in big blocks, apply OPQ rotation (R*X), compute SSE per gid using the
//     int8 coeff-codec runtime semantics, and write se_by_gid.f32 sequentially.

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <list>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <omp.h>

#include "stlq/pipeline/app_utils.h"
#include "stlq/common/config.h"
#include "stlq/core/kernel_provider_cuda_stream.h"
#include "stlq/eval/linkage_cluster_provider.h"
#include "stlq/io/linkage_coeff_codec_store.h"
#include "stlq/io/linkage_list_store.h"
#include "stlq/io/dataset_io.h"
#include "stlq/io/dataset_reader_factory.h"
#include "stlq/io/result_io.h"
#include "stlq/common/logger.h"
#include "stlq/quantizer/codebook_meta.h"
#include "stlq/common/timer.h"

namespace {

constexpr std::uint32_t kMissingCid = std::numeric_limits<std::uint32_t>::max();

struct Args {
    std::string run_root;
    std::string out_dir;
    std::uint64_t gid_bucket_size = 1000000ULL;
    int block_cols = 65536;
    std::string tmp_dir;
    int keep_tmp = 0;
    int write_se = 1;
    int keep_se = 1;
    int fast_dp = 0;
    std::uint64_t progress_clusters = 256ULL;
    int threads = 0;
    int use_cuda = -1;
    int strict_buckets = 0;
};

bool ParseBoolArg(const std::string& s, int* out) {
    if (!out) return false;
    if (s == "1" || s == "true" || s == "True" || s == "TRUE" || s == "yes") {
        *out = 1;
        return true;
    }
    if (s == "0" || s == "false" || s == "False" || s == "FALSE" || s == "no") {
        *out = 0;
        return true;
    }
    return false;
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
        auto take_int = [&](int* dst) -> bool {
            if (i + 1 >= argc) return false;
            *dst = std::stoi(argv[++i]);
            return true;
        };
        auto take_u64 = [&](std::uint64_t* dst) -> bool {
            if (i + 1 >= argc) return false;
            *dst = std::stoull(argv[++i]);
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
        } else if (a == "--gid_bucket_size") {
            if (!take_u64(&out->gid_bucket_size)) {
                if (err) *err = "Missing value for --gid_bucket_size";
                return false;
            }
        } else if (a == "--block_cols") {
            if (!take_int(&out->block_cols)) {
                if (err) *err = "Missing value for --block_cols";
                return false;
            }
        } else if (a == "--tmp_dir") {
            if (!take(&out->tmp_dir)) {
                if (err) *err = "Missing value for --tmp_dir";
                return false;
            }
        } else if (a == "--keep_tmp") {
            std::string v;
            if (!take(&v) || !ParseBoolArg(v, &out->keep_tmp)) {
                if (err) *err = "Invalid value for --keep_tmp";
                return false;
            }
        } else if (a == "--write_se") {
            std::string v;
            if (!take(&v) || !ParseBoolArg(v, &out->write_se)) {
                if (err) *err = "Invalid value for --write_se";
                return false;
            }
        } else if (a == "--keep_se") {
            std::string v;
            if (!take(&v) || !ParseBoolArg(v, &out->keep_se)) {
                if (err) *err = "Invalid value for --keep_se";
                return false;
            }
        } else if (a == "--fast_dp") {
            std::string v;
            if (!take(&v) || !ParseBoolArg(v, &out->fast_dp)) {
                if (err) *err = "Invalid value for --fast_dp";
                return false;
            }
        } else if (a == "--progress_clusters") {
            if (!take_u64(&out->progress_clusters)) {
                if (err) *err = "Missing value for --progress_clusters";
                return false;
            }
        } else if (a == "--threads") {
            if (!take_int(&out->threads)) {
                if (err) *err = "Missing value for --threads";
                return false;
            }
        } else if (a == "--use_cuda") {
            std::string v;
            if (!take(&v) || !ParseBoolArg(v, &out->use_cuda)) {
                if (err) *err = "Invalid value for --use_cuda";
                return false;
            }
        } else if (a == "--strict_buckets") {
            std::string v;
            if (!take(&v) || !ParseBoolArg(v, &out->strict_buckets)) {
                if (err) *err = "Invalid value for --strict_buckets";
                return false;
            }
        } else if (a == "-h" || a == "--help") {
            std::cout
                << "report_linkage_int8_recon_mse\n\n"
                << "Required:\n"
                << "  --run_root <path>   stlq run_root (contains config_snapshot.* and linkage_list/)\n\n"
                << "Optional:\n"
                << "  --out_dir <path>    Output dir (default: <run_root>/int8_recon_mse)\n"
                << "  --gid_bucket_size <u64>  Gid bucket size (default: 1000000)\n"
                << "  --block_cols <int>  Base vectors per read/rotate block (default: 65536)\n"
                << "  --tmp_dir <path>    Temp dir for bucket files (default: <out_dir>/tmp/report_linkage_int8_recon_mse)\n"
                << "  --keep_tmp <0|1>    Keep tmp_dir after completion (default: 0)\n"
                << "  --write_se <0|1>    Write se_by_gid.f32 (default: 1)\n"
                << "  --keep_se <0|1>     Keep se_by_gid.f32 after completion (default: 1; ignored if --write_se=0)\n"
                << "  --fast_dp <0|1>     Fast path: DP reconstruct vectors per gid and write tmp recon buckets (recon stored as fp16) (default: 0)\n"
                << "  --progress_clusters <u64>  Progress log stride in Phase1 cluster scan (default: 256)\n"
                << "  --threads <int>     OMP threads; 0 means config/runtime default\n"
                << "  --use_cuda <0|1>    Use CUDA for batch rotation when available (default: config/runtime.use_cuda)\n"
                << "  --strict_buckets <0|1>  Require every gid in each bucket to be present (default: 0)\n";
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
        out->out_dir = (std::filesystem::path(out->run_root) / "int8_recon_mse").string();
    }
    if (out->tmp_dir.empty()) {
        out->tmp_dir = (std::filesystem::path(out->out_dir) / "tmp" / "report_linkage_int8_recon_mse").string();
    }
    out->gid_bucket_size = std::max<std::uint64_t>(1ULL, out->gid_bucket_size);
    out->block_cols = std::max(1, out->block_cols);
    out->progress_clusters = std::max<std::uint64_t>(1ULL, out->progress_clusters);
    out->write_se = (out->write_se != 0) ? 1 : 0;
    out->keep_se = (out->keep_se != 0) ? 1 : 0;
    out->fast_dp = (out->fast_dp != 0) ? 1 : 0;
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

inline std::uint32_t ReadParent1BasedAt(const stlq::eval::ClusterView& cv, int pos) {
    if (pos < 0 || pos >= cv.n_real) {
        return 0u;
    }
    if (cv.parent_is_u16) {
        return cv.parent_1based_u16 ? static_cast<std::uint32_t>(cv.parent_1based_u16[pos]) : 0u;
    }
    return cv.parent_1based ? cv.parent_1based[pos] : 0u;
}

bool RotateBatchMaybeCuda(const stlq::ColMajorMatrix<float>& R,
                          bool use_cuda,
                          stlq::CudaStreamKernels* cuda,
                          stlq::ColMajorMatrix<float>* X,
                          double* out_sec,
                          std::string* err) {
    if (out_sec) *out_sec = 0.0;
    if (!X) return false;
    if (stlq::IsIdentityRotation(R)) return true;
    if (R.rows != R.cols || R.rows != X->rows) {
        if (err) *err = "RotateBatchMaybeCuda: dim mismatch.";
        return false;
    }
    const stlq::Timer t;
    if (use_cuda) {
        if (!cuda) {
            if (err) *err = "RotateBatchMaybeCuda: CUDA requested but kernel provider is null.";
            return false;
        }
        stlq::ColMajorMatrix<float> Xrot;
        cuda->Gemm(/*transA=*/false, /*transB=*/false, 1.0f, R, *X, 0.0f, &Xrot);
        cuda->Sync();
        *X = std::move(Xrot);
    } else {
        stlq::ApplyRotationInPlace(R, X);
    }
    if (out_sec) *out_sec = t.ElapsedSeconds();
    return true;
}

struct RunningStats {
    std::uint64_t count = 0;
    double sum = 0.0;
    double min = std::numeric_limits<double>::infinity();
    double max = 0.0;

    void Add(float v) {
        if (!std::isfinite(v)) return;
        count += 1;
        sum += static_cast<double>(v);
        min = std::min(min, static_cast<double>(v));
        max = std::max(max, static_cast<double>(v));
    }

    void Merge(const RunningStats& other) {
        if (other.count == 0) return;
        count += other.count;
        sum += other.sum;
        min = std::min(min, other.min);
        max = std::max(max, other.max);
    }
};

float DotF32(const float* a, const float* b, int d) {
    float s = 0.0f;
    #pragma omp simd reduction(+:s)
    for (int r = 0; r < d; ++r) {
        s += a[r] * b[r];
    }
    return s;
}

float ComputeSelfDotInt8(const stlq::eval::ClusterView& cv,
                         int cid,
                         int pos,
                         const float* x,
                         int d,
                         const stlq::ColMajorMatrix<float>& C_root0,
                         const stlq::CodebookMeta& meta_root_small,
                         const stlq::CodebookMeta& meta_one) {
    const int m = cv.m;
    const int n_virt = cv.n_virt;
    const int nc = cv.nc;
    const int stride_codes = cv.m_codes;

    auto q_i8 = [&](int layer, int local_idx) -> float {
        return static_cast<float>(
            cv.q_layer_major[static_cast<std::size_t>(layer) * static_cast<std::size_t>(nc) +
                             static_cast<std::size_t>(local_idx)]);
    };
    auto root_dot = [&](int root_local) -> float {
        float dot = cv.scales_root[0] * q_i8(0, root_local) * DotF32(C_root0.Col(cid), x, d);
        for (int l = 1; l < m; ++l) {
            const std::uint8_t code = (root_local < n_virt)
                ? cv.virt_codes_small_bytes[static_cast<std::size_t>(root_local) * static_cast<std::size_t>(stride_codes) +
                                            static_cast<std::size_t>(l - 1)]
                : cv.codes_small_bytes[static_cast<std::size_t>(root_local - n_virt) * static_cast<std::size_t>(stride_codes) +
                                       static_cast<std::size_t>(l - 1)];
            dot += cv.scales_root[l] * q_i8(l, root_local) *
                   DotF32(meta_root_small.flat.Col(meta_root_small.offsets[static_cast<std::size_t>(l - 1)] + static_cast<int>(code)),
                          x,
                          d);
        }
        return dot;
    };
    auto linkage_inc = [&](int local_idx, int real_pos) -> float {
        float dot = cv.scales_linkage[0] * q_i8(0, local_idx) *
                    DotF32(meta_one.flat.Col(meta_one.offsets[0] + static_cast<int>(cv.code0_one_bytes[real_pos])),
                           x,
                           d);
        const std::uint8_t* row =
            cv.codes_small_bytes + static_cast<std::size_t>(real_pos) * static_cast<std::size_t>(stride_codes);
        for (int l = 1; l < m; ++l) {
            dot += cv.scales_linkage[l] * q_i8(l, local_idx) *
                   DotF32(meta_one.flat.Col(meta_one.offsets[static_cast<std::size_t>(l)] + static_cast<int>(row[l - 1])),
                          x,
                          d);
        }
        return dot;
    };

    float dot = 0.0f;
    int local = n_virt + pos;
    while (true) {
        if (local < n_virt) {
            dot += root_dot(local);
            break;
        }
        const int cur_pos = local - n_virt;
        if (cur_pos < cv.n_root_real) {
            dot += root_dot(local);
            break;
        }
        dot += linkage_inc(local, cur_pos);
        const std::uint32_t p1 = ReadParent1BasedAt(cv, cur_pos);
        if (p1 == 0u) {
            dot += root_dot(local);
            break;
        }
        local = static_cast<int>(p1) - 1;
    }
    return dot;
}

float ReadSelfNorm2(const stlq::eval::ClusterView& cv, int pos) {
    if (cv.r_norm2) {
        return cv.r_norm2[pos];
    }
    if (cv.r_norm2_lut_u8 && cv.r_norm2_lut_centers && cv.r_norm2_lut_size > 0) {
        return cv.r_norm2_lut_centers[static_cast<int>(cv.r_norm2_lut_u8[pos])];
    }
    return 0.0f;
}

struct BucketRecord {
    std::uint32_t gid = 0;
    std::uint32_t cid = 0;
    std::uint32_t pos = 0;
};

// recon bucket record layout (binary):
//   u32 gid
//   f32 rnorm2
//   u16 recon_f16[d]   (IEEE 754 binary16 payload)
struct ReconRecordHdr {
    std::uint32_t gid = 0;
    float rnorm2 = 0.0f;
};

inline std::uint16_t FloatToF16Bits(float f) {
    std::uint32_t x = 0;
    std::memcpy(&x, &f, sizeof(x));
    const std::uint32_t sign = (x >> 16) & 0x8000u;
    const std::uint32_t mant = x & 0x007fffffu;
    const std::uint32_t exp = (x >> 23) & 0x000000ffu;

    if (exp == 255u) {
        // inf/NaN
        if (mant == 0u) return static_cast<std::uint16_t>(sign | 0x7c00u);
        return static_cast<std::uint16_t>(sign | 0x7e00u);
    }

    // Unbias exponent from f32 to f16.
    const int e = static_cast<int>(exp) - 127 + 15;
    if (e <= 0) {
        // subnormal or underflow to zero
        if (e < -10) return static_cast<std::uint16_t>(sign);
        std::uint32_t m = mant | 0x00800000u;
        const int shift = 14 - e;
        std::uint32_t h = m >> shift;
        // round to nearest even
        const std::uint32_t rem = m & ((1u << shift) - 1u);
        const std::uint32_t halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (h & 1u))) {
            h += 1u;
        }
        return static_cast<std::uint16_t>(sign | (h & 0x03ffu));
    }
    if (e >= 31) {
        // overflow to inf
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    }

    std::uint32_t h = sign | (static_cast<std::uint32_t>(e) << 10) | (mant >> 13);
    const std::uint32_t rem = mant & 0x00001fffu;
    if (rem > 0x00001000u || (rem == 0x00001000u && (h & 1u))) {
        h += 1u;
    }
    return static_cast<std::uint16_t>(h);
}

inline float F16BitsToFloat(std::uint16_t h) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(h & 0x8000u)) << 16;
    std::uint32_t exp = (h >> 10) & 0x001fu;
    std::uint32_t mant = static_cast<std::uint32_t>(h & 0x03ffu);

    std::uint32_t out = 0;
    if (exp == 0u) {
        if (mant == 0u) {
            out = sign;
        } else {
            // normalize subnormal
            exp = 1u;
            while ((mant & 0x0400u) == 0u) {
                mant <<= 1;
                exp -= 1u;
            }
            mant &= 0x03ffu;
            const std::uint32_t exp32 = (exp - 15u + 127u);
            out = sign | (exp32 << 23) | (mant << 13);
        }
    } else if (exp == 31u) {
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        const std::uint32_t exp32 = (exp - 15u + 127u);
        out = sign | (exp32 << 23) | (mant << 13);
    }
    float f = 0.0f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

std::string JoinPath(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    return (std::filesystem::path(a) / b).string();
}

bool EnsureDir(const std::string& dir, std::string* err) {
    try {
        std::filesystem::create_directories(dir);
        return true;
    } catch (...) {
        if (err) *err = "report_linkage_int8_recon_mse: failed to create dir: " + dir;
        return false;
    }
}

int DetermineMaxOpenBuckets(std::uint64_t nbucket) {
    int max_open = 256;
    if (max_open < 32) max_open = 32;
    if (nbucket < static_cast<std::uint64_t>(max_open)) {
        max_open = static_cast<int>(nbucket);
    }
    return std::max(1, max_open);
}

bool BuildGidBuckets(const stlq::io::LinkageListReader& linkage_list,
                     const std::filesystem::path& linkage_list_dir,
                     const std::string& tmp_dir,
                     std::uint64_t nbase,
                     std::uint64_t gid_bucket_size,
                     std::uint64_t progress_clusters,
                     std::uint64_t* out_records_written,
                     std::uint64_t* out_skipped_gid_out_of_range,
                     std::string* err) {
    if (out_records_written) *out_records_written = 0;
    if (out_skipped_gid_out_of_range) *out_skipped_gid_out_of_range = 0;
    if (gid_bucket_size == 0) {
        if (err) *err = "BuildGidBuckets: gid_bucket_size must be positive.";
        return false;
    }
    if (!EnsureDir(tmp_dir, err)) return false;

    const std::uint64_t nbucket = (nbase == 0) ? 0 : ((nbase + gid_bucket_size - 1) / gid_bucket_size);
    const int max_open_buckets = DetermineMaxOpenBuckets(nbucket);
    const std::size_t rec_bytes = sizeof(BucketRecord);
    const std::size_t flush_target = 4u * 1024u * 1024u;

    std::vector<std::ofstream> outs(static_cast<std::size_t>(nbucket));
    std::vector<bool> out_open(static_cast<std::size_t>(nbucket), false);
    std::vector<bool> out_created(static_cast<std::size_t>(nbucket), false);
    std::vector<std::vector<std::uint8_t>> bucket_bufs(static_cast<std::size_t>(nbucket));

    int open_count = 0;
    std::list<std::uint64_t> lru;
    std::vector<std::list<std::uint64_t>::iterator> lru_it(static_cast<std::size_t>(nbucket), lru.end());

    auto lru_touch = [&](std::uint64_t b) {
        auto& it = lru_it[static_cast<std::size_t>(b)];
        if (it == lru.end()) {
            lru.push_back(b);
            it = std::prev(lru.end());
        } else {
            lru.splice(lru.end(), lru, it);
        }
    };

    auto close_bucket = [&](std::uint64_t b) {
        const std::size_t bi = static_cast<std::size_t>(b);
        if (out_open[bi]) {
            if (outs[bi].is_open()) outs[bi].close();
            out_open[bi] = false;
            if (open_count > 0) --open_count;
        }
        auto& it = lru_it[bi];
        if (it != lru.end()) {
            lru.erase(it);
            it = lru.end();
        }
    };

    auto ensure_bucket = [&](std::uint64_t b) -> bool {
        const std::size_t bi = static_cast<std::size_t>(b);
        if (out_open[bi]) {
            lru_touch(b);
            return true;
        }
        while (open_count >= max_open_buckets && !lru.empty()) {
            close_bucket(lru.front());
        }
        const std::string path = JoinPath(tmp_dir, "gid_bucket_" + std::to_string(b) + ".bin");
        const auto mode = std::ios::binary | (out_created[bi] ? std::ios::app : std::ios::trunc);
        outs[bi].open(path, mode);
        if (!outs[bi].is_open()) {
            if (err) *err = "BuildGidBuckets: failed to open temp bucket: " + path;
            return false;
        }
        out_created[bi] = true;
        out_open[bi] = true;
        ++open_count;
        lru_touch(b);
        return true;
    };

    auto flush_bucket = [&](std::uint64_t b) -> bool {
        auto& buf = bucket_bufs[static_cast<std::size_t>(b)];
        if (buf.empty()) return true;
        if (!ensure_bucket(b)) return false;
        outs[static_cast<std::size_t>(b)].write(reinterpret_cast<const char*>(buf.data()),
                                                static_cast<std::streamsize>(buf.size()));
        if (!outs[static_cast<std::size_t>(b)]) {
            if (err) *err = "BuildGidBuckets: failed while writing temp bucket.";
            return false;
        }
        buf.clear();
        return true;
    };

    const std::filesystem::path real_ids_path = linkage_list_dir / "real_ids.u32";
    std::ifstream real_in(real_ids_path, std::ios::binary);
    if (!real_in.is_open()) {
        if (err) *err = "BuildGidBuckets: failed to open real_ids: " + real_ids_path.string();
        return false;
    }

    const auto& real_offsets = linkage_list.real_offsets();
    if (real_offsets.size() != static_cast<std::size_t>(linkage_list.nlist() + 1)) {
        if (err) *err = "BuildGidBuckets: unexpected real_offsets size.";
        return false;
    }

    std::vector<std::uint32_t> gids;
    BucketRecord rec{};
    std::uint64_t cur_real = 0;
    std::uint64_t written = 0;
    std::uint64_t skipped = 0;

    for (int cid = 0; cid < linkage_list.nlist(); ++cid) {
        const std::uint64_t lo = real_offsets[static_cast<std::size_t>(cid)];
        const std::uint64_t hi = real_offsets[static_cast<std::size_t>(cid + 1)];
        const std::uint64_t n_real = (hi >= lo) ? (hi - lo) : 0;
        if (lo != cur_real) {
            real_in.clear();
            real_in.seekg(static_cast<std::streamoff>(lo * sizeof(std::uint32_t)), std::ios::beg);
            cur_real = lo;
        }
        gids.resize(static_cast<std::size_t>(n_real));
        if (n_real > 0) {
            real_in.read(reinterpret_cast<char*>(gids.data()),
                         static_cast<std::streamsize>(n_real * sizeof(std::uint32_t)));
            if (!real_in) {
                if (err) *err = "BuildGidBuckets: failed while reading real_ids span.";
                return false;
            }
        }
        cur_real += n_real;

        for (std::uint32_t pos = 0; pos < static_cast<std::uint32_t>(n_real); ++pos) {
            const std::uint32_t gid = gids[static_cast<std::size_t>(pos)];
            if (gid >= nbase) {
                ++skipped;
                continue;
            }
            rec.gid = gid;
            rec.cid = static_cast<std::uint32_t>(cid);
            rec.pos = pos;
            const std::uint64_t b = static_cast<std::uint64_t>(gid) / gid_bucket_size;
            auto& buf = bucket_bufs[static_cast<std::size_t>(b)];
            const std::size_t old = buf.size();
            buf.resize(old + rec_bytes);
            std::memcpy(buf.data() + old, &rec, sizeof(rec));
            ++written;
            if (buf.size() >= flush_target) {
                if (!flush_bucket(b)) return false;
            }
        }

        if (progress_clusters > 0 && ((cid + 1) % static_cast<int>(progress_clusters) == 0 || cid + 1 == linkage_list.nlist())) {
            stlq::LogInfo("Phase1 bucketizing clusters " + std::to_string(cid + 1) + "/" +
                            std::to_string(linkage_list.nlist()));
        }
    }

    for (std::uint64_t b = 0; b < nbucket; ++b) {
        if (!flush_bucket(b)) return false;
        close_bucket(b);
    }

    if (out_records_written) *out_records_written = written;
    if (out_skipped_gid_out_of_range) *out_skipped_gid_out_of_range = skipped;
    return true;
}

inline void AddScaledVec(const float* src, int d, float scale, float* dst) {
    #pragma omp simd
    for (int i = 0; i < d; ++i) {
        dst[i] += scale * src[i];
    }
}

inline void CopyVec(const float* src, int d, float* dst) {
    std::memcpy(dst, src, static_cast<std::size_t>(d) * sizeof(float));
}

// Build per-gid recon buckets using per-cluster DP in parent-topological order.
// Writes temp files: recon_bucket_<b>.bin under tmp_dir.
bool BuildReconBucketsFastDpF16(stlq::eval::ClusterProvider* provider,
                                const stlq::io::LinkageListReader& linkage_list,
                                const std::filesystem::path& linkage_list_dir,
                                const stlq::ColMajorMatrix<float>& C_root0,
                                const stlq::CodebookMeta& meta_root_small,
                                const stlq::CodebookMeta& meta_one,
                                int d,
                                const std::string& tmp_dir,
                                std::uint64_t nbase,
                                std::uint64_t gid_bucket_size,
                                std::uint64_t progress_clusters,
                                std::uint64_t* out_records_written,
                                std::uint64_t* out_skipped_gid_out_of_range,
                                std::string* err) {
    if (!provider) {
        if (err) *err = "BuildReconBucketsFastDpF16: provider is null.";
        return false;
    }
    if (out_records_written) *out_records_written = 0;
    if (out_skipped_gid_out_of_range) *out_skipped_gid_out_of_range = 0;
    if (gid_bucket_size == 0) {
        if (err) *err = "BuildReconBucketsFastDpF16: gid_bucket_size must be positive.";
        return false;
    }
    if (d <= 0) {
        if (err) *err = "BuildReconBucketsFastDpF16: invalid d.";
        return false;
    }
    (void)linkage_list_dir;
    if (!EnsureDir(tmp_dir, err)) return false;

    const std::uint64_t nbucket = (nbase == 0) ? 0 : ((nbase + gid_bucket_size - 1) / gid_bucket_size);
    const std::size_t rec_bytes = sizeof(ReconRecordHdr) + static_cast<std::size_t>(d) * sizeof(std::uint16_t);
    const std::size_t flush_target = 8u * 1024u * 1024u;
    const int max_active_buckets = std::max(1, std::min<int>(16, static_cast<int>(nbucket)));

    auto part_path = [&](std::uint64_t b, int tid) {
        return JoinPath(tmp_dir, "recon_bucket_" + std::to_string(b) + "_t" + std::to_string(tid) + "_f16.bin");
    };

    // Build stable per-cid views in serial (GetCluster is not assumed thread-safe).
    std::vector<stlq::eval::ClusterView> cv_by_cid(static_cast<std::size_t>(linkage_list.nlist()));
    {
        for (int cid = 0; cid < linkage_list.nlist(); ++cid) {
            stlq::eval::ClusterView cv;
            stlq::eval::ClusterProvider::PrepStats prep{};
            std::string local_err;
            if (!provider->GetCluster(cid, &cv, &prep, &local_err)) {
                if (err) *err = "BuildReconBucketsFastDpF16: GetCluster failed for cid=" + std::to_string(cid) + ": " + local_err;
                return false;
            }
            cv_by_cid[static_cast<std::size_t>(cid)] = cv;
        }
    }

    std::uint64_t written = 0;
    std::uint64_t skipped = 0;
    std::atomic<bool> ok(true);
    std::string first_err;

    // Parallel over clusters.
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();

        std::vector<int> parent_local;
        std::vector<std::uint8_t> state;
        std::vector<float> v_local;
        std::vector<int> stack;
        std::uint64_t local_written = 0;
        std::uint64_t local_skipped = 0;
        std::string local_err;

        struct ActiveBucket {
            std::uint64_t b = 0;
            std::uint64_t last_use = 0;
            std::vector<std::uint8_t> buf;
            std::ofstream out;
            bool opened = false;
        };
        std::vector<ActiveBucket> active;
        active.reserve(static_cast<std::size_t>(max_active_buckets));
        std::uint64_t tick = 0;

        auto flush_entry = [&](ActiveBucket* e) -> bool {
            if (!e) return false;
            if (e->buf.empty()) return true;
            if (!e->opened) {
                const std::string path = part_path(e->b, tid);
                e->out.open(path, std::ios::binary | std::ios::out | std::ios::app);
                if (!e->out.is_open()) {
                    local_err = "BuildReconBucketsFastDpF16: failed to open temp recon bucket part for append: " + path;
                    return false;
                }
                e->opened = true;
            }
            e->out.write(reinterpret_cast<const char*>(e->buf.data()), static_cast<std::streamsize>(e->buf.size()));
            if (!e->out) {
                local_err = "BuildReconBucketsFastDpF16: failed while writing temp recon bucket part.";
                return false;
            }
            e->buf.clear();
            return true;
        };

        auto close_entry = [&](ActiveBucket* e) {
            if (!e) return;
            if (e->opened) {
                if (e->out.is_open()) e->out.close();
                e->opened = false;
            }
        };

        auto get_entry = [&](std::uint64_t b) -> ActiveBucket* {
            ++tick;
            for (auto& e : active) {
                if (e.b == b) {
                    e.last_use = tick;
                    return &e;
                }
            }
            if (static_cast<int>(active.size()) < max_active_buckets) {
                active.push_back({});
                active.back().b = b;
                active.back().last_use = tick;
                return &active.back();
            }
            // evict LRU
            std::size_t ev = 0;
            for (std::size_t i = 1; i < active.size(); ++i) {
                if (active[i].last_use < active[ev].last_use) ev = i;
            }
            ActiveBucket* e = &active[ev];
            if (!flush_entry(e)) return nullptr;
            close_entry(e);
            e->b = b;
            e->last_use = tick;
            e->buf.clear();
            return e;
        };

        auto append_record = [&](std::uint64_t b, std::uint32_t gid, float rnorm2, const float* vec) -> bool {
            ActiveBucket* e = get_entry(b);
            if (!e) return false;
            if (e->buf.size() + rec_bytes > flush_target) {
                if (!flush_entry(e)) return false;
            }
            const std::size_t old = e->buf.size();
            e->buf.resize(old + rec_bytes);
            ReconRecordHdr hdr{};
            hdr.gid = gid;
            hdr.rnorm2 = rnorm2;
            std::memcpy(e->buf.data() + old, &hdr, sizeof(hdr));
            std::uint8_t* dst = e->buf.data() + old + sizeof(hdr);
            for (int i = 0; i < d; ++i) {
                const std::uint16_t hb = FloatToF16Bits(vec[i]);
                std::memcpy(dst + static_cast<std::size_t>(i) * sizeof(std::uint16_t), &hb, sizeof(hb));
            }
            if (e->buf.size() >= flush_target) {
                if (!flush_entry(e)) return false;
            }
            return true;
        };

        #pragma omp for schedule(dynamic, 1)
        for (int cid = 0; cid < linkage_list.nlist(); ++cid) {
            if (!ok.load(std::memory_order_relaxed)) continue;
            const auto& cv = cv_by_cid[static_cast<std::size_t>(cid)];
        if (cv.m <= 0 || cv.nc <= 0) {
                continue;
        }
        if (d != C_root0.rows) {
                continue;
        }

        const int n_virt = cv.n_virt;
        const int n_real = cv.n_real;
        const int n_local = n_virt + n_real;
        if (n_real < 0 || n_local <= 0) {
            continue;
        }

        parent_local.assign(static_cast<std::size_t>(n_local), -1);
        for (int local = 0; local < n_local; ++local) {
            if (local < n_virt) {
                parent_local[static_cast<std::size_t>(local)] = -1;
                continue;
            }
            const int pos = local - n_virt;
            if (pos < cv.n_root_real) {
                parent_local[static_cast<std::size_t>(local)] = -1;
                continue;
            }
            const std::uint32_t p1 = ReadParent1BasedAt(cv, pos);
            if (p1 == 0u) {
                parent_local[static_cast<std::size_t>(local)] = -1;
                continue;
            }
            parent_local[static_cast<std::size_t>(local)] = static_cast<int>(p1) - 1;
        }

        v_local.assign(static_cast<std::size_t>(n_local) * static_cast<std::size_t>(d), 0.0f);
        state.assign(static_cast<std::size_t>(n_local), 0u);
        stack.clear();
        stack.reserve(256);

        auto q_i8 = [&](int layer, int local_idx) -> float {
            return static_cast<float>(
                cv.q_layer_major[static_cast<std::size_t>(layer) * static_cast<std::size_t>(cv.nc) +
                                 static_cast<std::size_t>(local_idx)]);
        };

        auto compute_root_vec = [&](int local_idx, float* dst) {
            std::fill(dst, dst + d, 0.0f);
            const float s0 = cv.scales_root[0] * q_i8(0, local_idx);
            AddScaledVec(C_root0.Col(cid), d, s0, dst);
            for (int l = 1; l < cv.m; ++l) {
                const std::uint8_t code = (local_idx < n_virt)
                    ? cv.virt_codes_small_bytes[static_cast<std::size_t>(local_idx) * static_cast<std::size_t>(cv.m_codes) +
                                                static_cast<std::size_t>(l - 1)]
                    : cv.codes_small_bytes[static_cast<std::size_t>(local_idx - n_virt) * static_cast<std::size_t>(cv.m_codes) +
                                           static_cast<std::size_t>(l - 1)];
                const float s = cv.scales_root[l] * q_i8(l, local_idx);
                const int col = meta_root_small.offsets[static_cast<std::size_t>(l - 1)] + static_cast<int>(code);
                AddScaledVec(meta_root_small.flat.Col(col), d, s, dst);
            }
        };

        auto add_linkage_inc = [&](int local_idx, int pos, float* dst) {
            const float s0 = cv.scales_linkage[0] * q_i8(0, local_idx);
            const int col0 = meta_one.offsets[0] + static_cast<int>(cv.code0_one_bytes[pos]);
            AddScaledVec(meta_one.flat.Col(col0), d, s0, dst);
            const std::uint8_t* row = cv.codes_small_bytes + static_cast<std::size_t>(pos) * static_cast<std::size_t>(cv.m_codes);
            for (int l = 1; l < cv.m; ++l) {
                const float s = cv.scales_linkage[l] * q_i8(l, local_idx);
                const int col = meta_one.offsets[static_cast<std::size_t>(l)] + static_cast<int>(row[l - 1]);
                AddScaledVec(meta_one.flat.Col(col), d, s, dst);
            }
        };

        auto compute_node = [&](int start_local) -> bool {
            int cur = start_local;
            while (true) {
                const std::uint8_t st = state[static_cast<std::size_t>(cur)];
                if (st == 2u) {
                    break;
                }
                if (st == 1u) {
                    local_err = "BuildReconBucketsFastDpF16: cycle detected in parent pointers.";
                    return false;
                }
                state[static_cast<std::size_t>(cur)] = 1u;
                stack.push_back(cur);
                const int p = parent_local[static_cast<std::size_t>(cur)];
                if (p < 0) {
                    break;
                }
                cur = p;
            }
            while (!stack.empty()) {
                const int node = stack.back();
                stack.pop_back();
                const int p = parent_local[static_cast<std::size_t>(node)];
                float* dst = v_local.data() + static_cast<std::size_t>(node) * static_cast<std::size_t>(d);
                if (p < 0) {
                    compute_root_vec(node, dst);
                } else {
                    const float* src = v_local.data() + static_cast<std::size_t>(p) * static_cast<std::size_t>(d);
                    CopyVec(src, d, dst);
                    add_linkage_inc(node, node - n_virt, dst);
                }
                state[static_cast<std::size_t>(node)] = 2u;
            }
            return true;
        };

        for (int local = 0; local < n_local; ++local) {
            if (state[static_cast<std::size_t>(local)] == 2u) continue;
                if (!compute_node(local)) {
                    ok.store(false, std::memory_order_relaxed);
                    break;
                }
        }
            if (!ok.load(std::memory_order_relaxed)) {
                #pragma omp critical
                {
                    if (first_err.empty()) first_err = local_err;
                }
                continue;
            }

        if (!cv.real_ids) {
            continue;
        }

        // HDD-friendly write pattern: sort by gid so bucket id is near-monotonic
        // within each cluster. This reduces LRU thrash and random seeks when nbucket is large.
        std::vector<std::pair<std::uint32_t, int>> gid_pos;
        gid_pos.reserve(static_cast<std::size_t>(std::max(0, n_real)));
        for (int pos = 0; pos < n_real; ++pos) {
            const std::uint32_t gid = cv.real_ids[static_cast<std::size_t>(pos)];
            if (gid >= nbase) {
                ++local_skipped;
                continue;
            }
            gid_pos.emplace_back(gid, pos);
        }
        std::sort(gid_pos.begin(), gid_pos.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        for (const auto& gp : gid_pos) {
            const std::uint32_t gid = gp.first;
            const int pos = gp.second;
            const float rnorm2 = ReadSelfNorm2(cv, pos);
            const int local_idx = n_virt + pos;
            const float* vec = v_local.data() + static_cast<std::size_t>(local_idx) * static_cast<std::size_t>(d);
            const std::uint64_t b = static_cast<std::uint64_t>(gid) / gid_bucket_size;
            if (b >= nbucket) {
                ++local_skipped;
                continue;
            }
            if (!append_record(b, gid, rnorm2, vec)) {
                ok.store(false, std::memory_order_relaxed);
                #pragma omp critical
                {
                    if (first_err.empty()) first_err = local_err;
                }
                break;
            }
            ++local_written;
        }

            if (progress_clusters > 0 && ((cid + 1) % static_cast<int>(progress_clusters) == 0 || cid + 1 == linkage_list.nlist())) {
                #pragma omp critical
                {
                    stlq::LogInfo("Phase1 fast_dp reconstructed clusters " + std::to_string(cid + 1) + "/" +
                                    std::to_string(linkage_list.nlist()));
                }
            }
        }

        // Flush remaining buffers for this thread.
        for (auto& e : active) {
            if (!flush_entry(&e)) {
                ok.store(false, std::memory_order_relaxed);
                #pragma omp critical
                {
                    if (first_err.empty()) first_err = local_err;
                }
                break;
            }
            close_entry(&e);
        }

        #pragma omp atomic
        written += local_written;
        #pragma omp atomic
        skipped += local_skipped;
    }

    if (!ok.load(std::memory_order_relaxed)) {
        if (err) *err = first_err.empty() ? "BuildReconBucketsFastDpF16: failed in parallel DP build." : first_err;
        return false;
    }

    if (out_records_written) *out_records_written = written;
    if (out_skipped_gid_out_of_range) *out_skipped_gid_out_of_range = skipped;
    return true;
}

bool LoadReconBucketAny(const std::string& tmp_dir,
                        std::uint64_t b,
                        std::uint64_t g0,
                        std::uint64_t bucket_len,
                        int d,
                        bool strict,
                        std::vector<float>* rnorm2_by_local,
                        std::vector<float>* recon_by_local,
                        std::uint64_t* out_seen,
                        std::string* err);

static bool LoadReconBucketFromFileInto(const std::string& path,
                                        std::uint64_t g0,
                                        std::uint64_t bucket_len,
                                        int d,
                                        std::vector<float>* rnorm2_by_local,
                                        std::vector<float>* recon_by_local,
                                        std::vector<std::uint8_t>* seen_flag,
                                        std::uint64_t* io_seen,
                                        std::string* err) {
    if (!rnorm2_by_local || !recon_by_local || !seen_flag || !io_seen) {
        if (err) *err = "LoadReconBucketFromFileInto: null output.";
        return false;
    }
    if (bucket_len == 0) return true;
    if (d <= 0) {
        if (err) *err = "LoadReconBucketFromFileInto: invalid d.";
        return false;
    }
    std::error_code ec;
    const std::size_t rec_bytes_f16 = sizeof(ReconRecordHdr) + static_cast<std::size_t>(d) * sizeof(std::uint16_t);
    const std::size_t rec_bytes_f32 = sizeof(ReconRecordHdr) + static_cast<std::size_t>(d) * sizeof(float);
    const std::uint64_t fsz = static_cast<std::uint64_t>(std::filesystem::file_size(path, ec));
    if (ec) {
        if (ec == std::errc::no_such_file_or_directory) {
            return true;  // missing part is not an error; strict mode is checked by seen count.
        }
        if (err) *err = "LoadReconBucketFromFileInto: file_size failed: " + path;
        return false;
    }
    const bool ok_f16 = (rec_bytes_f16 > 0) && ((fsz % static_cast<std::uint64_t>(rec_bytes_f16)) == 0ULL);
    const bool ok_f32 = (rec_bytes_f32 > 0) && ((fsz % static_cast<std::uint64_t>(rec_bytes_f32)) == 0ULL);
    const bool use_f16 = ok_f16 && (!ok_f32 || path.find("_f16") != std::string::npos);
    const std::size_t rec_bytes = use_f16 ? rec_bytes_f16 : rec_bytes_f32;
    if (!ok_f16 && !ok_f32) {
        if (err) *err = "LoadReconBucketFromFileInto: corrupted temp recon bucket (size not divisible by record size): " + path;
        return false;
    }

    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        if (err) *err = "LoadReconBucketFromFileInto: failed to open: " + path;
        return false;
    }

    std::vector<std::uint8_t> buf(rec_bytes * 4096u);
    while (true) {
        in.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
        const std::streamsize got = in.gcount();
        if (got <= 0) break;
        const std::size_t bytes = static_cast<std::size_t>(got);
        const std::size_t nrec = bytes / rec_bytes;
        const std::uint8_t* p = buf.data();
        for (std::size_t i = 0; i < nrec; ++i, p += rec_bytes) {
            ReconRecordHdr hdr{};
            std::memcpy(&hdr, p, sizeof(hdr));
            if (hdr.gid < g0 || hdr.gid >= g0 + bucket_len) {
                if (err) *err = "LoadReconBucketFromFileInto: gid outside bucket range.";
                return false;
            }
            const std::size_t local = static_cast<std::size_t>(hdr.gid - g0);
            if ((*seen_flag)[local]) {
                if (err) *err = "LoadReconBucketFromFileInto: duplicate gid across recon bucket parts.";
                return false;
            }
            (*seen_flag)[local] = 1u;
            (*rnorm2_by_local)[local] = hdr.rnorm2;
            float* dst = recon_by_local->data() + local * static_cast<std::size_t>(d);
            if (!use_f16) {
                std::memcpy(dst, p + sizeof(hdr), static_cast<std::size_t>(d) * sizeof(float));
            } else {
                const std::uint8_t* src = p + sizeof(hdr);
                for (int j = 0; j < d; ++j) {
                    std::uint16_t hb = 0;
                    std::memcpy(&hb, src + static_cast<std::size_t>(j) * sizeof(std::uint16_t), sizeof(hb));
                    dst[j] = F16BitsToFloat(hb);
                }
            }
            ++(*io_seen);
        }
    }
    return true;
}

bool LoadReconBucketAny(const std::string& tmp_dir,
                        std::uint64_t b,
                        std::uint64_t g0,
                        std::uint64_t bucket_len,
                        int d,
                        bool strict,
                        std::vector<float>* rnorm2_by_local,
                        std::vector<float>* recon_by_local,
                        std::uint64_t* out_seen,
                        std::string* err) {
    if (!rnorm2_by_local || !recon_by_local) {
        if (err) *err = "LoadReconBucketAny: null output.";
        return false;
    }
    rnorm2_by_local->assign(static_cast<std::size_t>(bucket_len), std::numeric_limits<float>::quiet_NaN());
    recon_by_local->assign(static_cast<std::size_t>(bucket_len) * static_cast<std::size_t>(std::max(0, d)), 0.0f);
    if (out_seen) *out_seen = 0;
    if (bucket_len == 0) return true;

    std::vector<std::uint8_t> seen_flag(static_cast<std::size_t>(bucket_len), 0u);
    std::uint64_t seen = 0;

    // Prefer legacy single-file format if present.
    {
        const std::string p0 = JoinPath(tmp_dir, "recon_bucket_" + std::to_string(b) + ".bin");
        std::error_code ec;
        if (std::filesystem::exists(p0, ec) && !ec) {
            if (!LoadReconBucketFromFileInto(p0, g0, bucket_len, d,
                                             rnorm2_by_local, recon_by_local,
                                             &seen_flag, &seen, err)) {
                return false;
            }
            if (strict && seen != bucket_len) {
                if (err) *err = "LoadReconBucketAny: incomplete bucket (strict) for legacy file.";
                return false;
            }
            if (out_seen) *out_seen = seen;
            return true;
        }
    }

    const int max_t = std::max(1, omp_get_max_threads());
    for (int t = 0; t < max_t; ++t) {
        {
            const std::string pt16 = JoinPath(tmp_dir, "recon_bucket_" + std::to_string(b) + "_t" + std::to_string(t) + "_f16.bin");
            if (!LoadReconBucketFromFileInto(pt16, g0, bucket_len, d,
                                             rnorm2_by_local, recon_by_local,
                                             &seen_flag, &seen, err)) {
                return false;
            }
        }
        const std::string pt = JoinPath(tmp_dir, "recon_bucket_" + std::to_string(b) + "_t" + std::to_string(t) + ".bin");
        if (!LoadReconBucketFromFileInto(pt, g0, bucket_len, d,
                                         rnorm2_by_local, recon_by_local,
                                         &seen_flag, &seen, err)) {
            return false;
        }
    }

    if (strict && seen != bucket_len) {
        if (err) {
            *err = "LoadReconBucketAny: incomplete bucket (strict), expected " + std::to_string(bucket_len) +
                   " records, got " + std::to_string(seen) + ".";
        }
        return false;
    }
    if (out_seen) *out_seen = seen;
    return true;
}

bool LoadBucket(const std::string& path,
                std::uint64_t g0,
                std::uint64_t bucket_len,
                bool strict,
                std::vector<std::uint32_t>* cid_by_local,
                std::vector<std::uint32_t>* pos_by_local,
                std::uint64_t* out_seen,
                std::string* err) {
    if (!cid_by_local || !pos_by_local) {
        if (err) *err = "LoadBucket: null output.";
        return false;
    }
    cid_by_local->assign(static_cast<std::size_t>(bucket_len), kMissingCid);
    pos_by_local->assign(static_cast<std::size_t>(bucket_len), 0);
    if (out_seen) *out_seen = 0;
    if (bucket_len == 0) return true;

    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || ec) {
        if (strict) {
            if (err) *err = "LoadBucket: missing temp bucket (strict): " + path;
            return false;
        }
        return true;
    }

    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        if (err) *err = "LoadBucket: failed to open temp bucket: " + path;
        return false;
    }
    const std::size_t rec_bytes = sizeof(BucketRecord);
    std::vector<std::uint8_t> buf(rec_bytes * 65536u);
    std::uint64_t seen = 0;

    while (true) {
        in.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
        const std::streamsize got = in.gcount();
        if (got <= 0) break;
        const std::size_t bytes = static_cast<std::size_t>(got);
        const std::size_t nrec = bytes / rec_bytes;
        const std::uint8_t* p = buf.data();
        for (std::size_t i = 0; i < nrec; ++i, p += rec_bytes) {
            BucketRecord rec{};
            std::memcpy(&rec, p, sizeof(rec));
            if (rec.gid < g0 || rec.gid >= g0 + bucket_len) {
                if (err) *err = "LoadBucket: gid outside bucket range.";
                return false;
            }
            const std::size_t local = static_cast<std::size_t>(rec.gid - g0);
            if ((*cid_by_local)[local] != kMissingCid) {
                if (err) *err = "LoadBucket: duplicate gid in temp bucket.";
                return false;
            }
            (*cid_by_local)[local] = rec.cid;
            (*pos_by_local)[local] = rec.pos;
            ++seen;
        }
    }

    if (strict && seen != bucket_len) {
        if (err) {
            *err = "LoadBucket: incomplete bucket (strict), expected " + std::to_string(bucket_len) +
                   " records, got " + std::to_string(seen) + ".";
        }
        return false;
    }
    if (out_seen) *out_seen = seen;
    return true;
}

std::string FormatDouble(double v, int prec) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(prec) << v;
    return oss.str();
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

    if (args.threads > 0) {
        cfg.runtime.omp_threads = args.threads;
    }
    omp_set_num_threads(std::max(1, cfg.runtime.omp_threads));

    bool use_cuda = (args.use_cuda >= 0) ? (args.use_cuda != 0) : cfg.runtime.use_cuda;

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
            LogInfo("Using run_state effective_train_h5: " + effective_train_h5);
        }
        base_path = ResolveLoadBasePath(base_path, load_date, load_seq, run_root, cfg_path);
        if (!io::LoadTrainResults(base_path, load_date, load_seq, cfg.model.m, &train, &train_err)) {
            LogError(train_err.empty() ? "LoadTrainResults failed." : train_err);
            return 1;
        }
    }

    io::LinkageListReader linkage_list;
    const std::filesystem::path linkage_list_dir = run_root / "linkage_list";
    if (!linkage_list.Open(linkage_list_dir.string(), &err)) {
        LogError(err);
        return 1;
    }

    io::LinkageCoeffCodecReader coeff_reader;
    if (!coeff_reader.Open(linkage_list_dir.string(), &err)) {
        LogError("Missing or invalid coeff codec store under linkage_list/: " + err);
        return 1;
    }

    stlq::io::DatasetVectorReader base_r;
    if (!stlq::io::OpenDatasetVectorReader(cfg, stlq::io::DatasetRole::kBase, &base_r, &err)) {
        LogError(err);
        return 1;
    }

    const std::filesystem::path norm2_direct_f32 = linkage_list_dir / "norm2_int8.f32";
    const std::filesystem::path norm2_lut_codes_u8 = linkage_list_dir / "norm2_int8_lut_codes.u8";
    const std::filesystem::path norm2_lut_centers_f32 = linkage_list_dir / "norm2_int8_lut_centers.f32";
    const bool has_norm2_direct_f32 = std::filesystem::exists(norm2_direct_f32);
    const bool has_norm2_lut =
        std::filesystem::exists(norm2_lut_codes_u8) && std::filesystem::exists(norm2_lut_centers_f32);
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

    eval::ClusterProvider provider;
    if (!provider.Open(linkage_list,
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
                       &err)) {
        LogError(err.empty() ? "ClusterProvider::Open failed." : err);
        return 1;
    }
    if (use_norm2_lut_global) {
        std::ifstream in(norm2_lut_centers_f32, std::ios::binary | std::ios::ate);
        if (in.is_open()) {
            const std::size_t sz = static_cast<std::size_t>(in.tellg());
            in.seekg(0, std::ios::beg);
            std::vector<float> centers(sz / sizeof(float), 0.0f);
            in.read(reinterpret_cast<char*>(centers.data()), static_cast<std::streamsize>(sz));
            if (in) {
                provider.SetGlobalNorm2LutCenters(std::move(centers));
            }
        }
    }
    provider.SetNorm2DiskLazyEnabled(has_norm2_direct_f32 || use_norm2_lut);

    const ColMajorMatrix<float>& C_root0 = train.C_root.books.front();
    std::vector<const ColMajorMatrix<float>*> root_small_books;
    root_small_books.reserve(static_cast<std::size_t>(std::max(0, cfg.model.m - 1)));
    for (int l = 1; l < cfg.model.m; ++l) {
        root_small_books.push_back(&train.C_root.books[static_cast<std::size_t>(l)]);
    }
    const CodebookMeta meta_root_small = BuildCodebookMeta(root_small_books);
    const CodebookMeta meta_one = BuildCodebookMeta(GatherBooks(train.C_one));

    const std::uint64_t nbase_cfg = static_cast<std::uint64_t>(std::max(0, cfg.dataset.nbase));
    const std::uint64_t base_n = static_cast<std::uint64_t>(base_r.n);
    std::uint64_t nbase = nbase_cfg;
    const std::uint64_t total_real = linkage_list.total_real();
    if (nbase == 0) {
        LogError("dataset.nbase is zero; cannot size output array.");
        return 1;
    }
    if (base_r.d != C_root0.rows) {
        LogError("Base dim mismatch: base_d=" + std::to_string(base_r.d) +
                 " codebook_d=" + std::to_string(C_root0.rows));
        return 1;
    }
    // Some run_roots may carry an inconsistent dataset.nbase in the snapshot (e.g. 1e9)
    // while the actual base file is smaller (e.g. DEEP1M with 1e6 vectors). For this tool,
    // we only can scan what exists in the base file; clamp to avoid huge allocations.
    if (base_n < nbase) {
        LogInfo("[WARN] dataset.nbase=" + std::to_string(nbase_cfg) +
                " but base file has base_n=" + std::to_string(base_n) +
                "; clamping nbase to base_n for analysis.");
        nbase = base_n;
    }
    if (total_real > nbase) {
        LogError("LinkageList real count exceeds effective nbase: total_real=" + std::to_string(total_real) +
                 " nbase=" + std::to_string(nbase));
        return 1;
    }

    std::error_code ec;
    std::filesystem::create_directories(args.out_dir, ec);
    if (ec) {
        LogError("Failed to create out_dir: " + args.out_dir);
        return 1;
    }

    // Prepare tmp dir.
    {
        std::error_code ec2;
        std::filesystem::remove_all(args.tmp_dir, ec2);
        if (!EnsureDir(args.tmp_dir, &err)) {
            LogError(err);
            return 1;
        }
    }

    if (args.fast_dp) {
        // Fast path writes large tmp recon buckets: {u32 gid, f32 rnorm2, f16 recon[d]}.
        const std::uint64_t rec_bytes = static_cast<std::uint64_t>(sizeof(ReconRecordHdr)) +
                                        static_cast<std::uint64_t>(2ULL * static_cast<std::uint64_t>(C_root0.rows));
        const long double est_bytes = static_cast<long double>(total_real) * static_cast<long double>(rec_bytes);
        std::error_code ec_sp;
        const auto sp = std::filesystem::space(args.tmp_dir, ec_sp);
        if (!ec_sp) {
            const long double avail = static_cast<long double>(sp.available);
            if (est_bytes > avail * 0.95L) {
                LogError("fast_dp requires ~" + FormatDouble(static_cast<double>(est_bytes / (1024.0L * 1024.0L * 1024.0L)), 2) +
                         " GiB tmp space but only " +
                         FormatDouble(static_cast<double>(avail / (1024.0L * 1024.0L * 1024.0L)), 2) +
                         " GiB is available under tmp_dir=" + args.tmp_dir);
                return 1;
            }
            if (est_bytes > 64.0L * 1024.0L * 1024.0L * 1024.0L) {
                LogInfo("[WARN] fast_dp will write ~" +
                        FormatDouble(static_cast<double>(est_bytes / (1024.0L * 1024.0L * 1024.0L)), 2) +
                        " GiB tmp recon data under tmp_dir=" + args.tmp_dir);
            }
        }
    }

    const std::filesystem::path se_path = std::filesystem::path(args.out_dir) / "stlq_int8_recon_se_by_gid.f32";
    std::ofstream se_out;
    const bool write_se = (args.write_se != 0);
    const bool keep_se = (args.keep_se != 0);
    if (write_se) {
        se_out.open(se_path, std::ios::binary | std::ios::out | std::ios::trunc);
        if (!se_out.is_open()) {
            LogError("Failed to open output file: " + se_path.string());
            return 1;
        }
    }

#if STLQ_ENABLE_CUDA
    std::unique_ptr<CudaStreamKernels> cuda_kernels;
    if (use_cuda) {
        try {
            cuda_kernels = std::make_unique<CudaStreamKernels>(cfg.runtime.cuda_device,
                                                               cfg.runtime.cuda_allow_tf32,
                                                               cfg.runtime.cuda_cublas_workspace_mb);
        } catch (const std::exception& e) {
            LogError(std::string("Failed to initialize CUDA kernels: ") + e.what());
            return 1;
        }
    }
#else
    if (use_cuda) {
        if (args.use_cuda >= 0 && args.use_cuda != 0) {
            LogError("This binary was built without CUDA support, but --use_cuda=1 was requested.");
            return 1;
        }
        LogInfo("[WARN] This binary was built without CUDA; config/runtime.use_cuda=1 is ignored. Falling back to CPU rotation.");
        use_cuda = false;
    }
    CudaStreamKernels* cuda_kernels = nullptr;
#endif
    CudaStreamKernels* cuda_ptr =
#if STLQ_ENABLE_CUDA
        cuda_kernels.get();
#else
        cuda_kernels;
#endif

    RunningStats stats;
    std::uint64_t neg_se_count = 0;
    double min_se_raw = std::numeric_limits<double>::infinity();
    std::uint64_t written_real = 0;
    std::uint64_t recon_failures = 0;
    std::uint64_t missing_records = 0;
    std::uint64_t skipped_gid_out_of_range = 0;
    std::uint64_t bucket_records_written = 0;
    double bucketize_sec = 0.0;
    double fast_dp_build_sec = 0.0;
    double preload_sec = 0.0;
    double cluster_view_init_sec = 0.0;
    double total_base_read_sec = 0.0;
    double total_rotate_sec = 0.0;
    double total_recon_sec = 0.0;
    double total_bucket_load_sec = 0.0;
    double total_recon_bucket_load_sec = 0.0;
    const Timer wall;

    eval::ClusterProvider::PreloadStats preload_stats{};
    if (args.fast_dp && cfg.eval.linkage_preload_clusters_io_threads >= 0) {
        const Timer t;
        LogInfo("Phase0: preloading all cluster data for fast_dp (io_threads=" +
                std::to_string(cfg.eval.linkage_preload_clusters_io_threads) + ") ...");
        if (!provider.PreloadAllClusters(cfg.eval.linkage_preload_clusters_io_threads, &preload_stats, &err)) {
            LogError(err.empty() ? "PreloadAllClusters failed." : err);
            return 1;
        }
        preload_sec = t.ElapsedSeconds();
        provider.ReleaseCoeffRawPayloads();
    }

    if (args.fast_dp) {
        const Timer t;
        LogInfo("Phase1 fast_dp: DP reconstruct vectors and write recon buckets (fp16) ...");
        const int d = C_root0.rows;
        if (!BuildReconBucketsFastDpF16(&provider,
                                        linkage_list,
                                        linkage_list_dir,
                                        C_root0,
                                        meta_root_small,
                                        meta_one,
                                        d,
                                        args.tmp_dir,
                                        nbase,
                                        args.gid_bucket_size,
                                        args.progress_clusters,
                                        &bucket_records_written,
                                        &skipped_gid_out_of_range,
                                        &err)) {
            LogError(err);
            return 1;
        }
        fast_dp_build_sec = t.ElapsedSeconds();
        provider.ReleaseCoeffRawPayloads();
    } else {
        const Timer t;
        LogInfo("Phase1: bucketizing linkage_list real_ids by gid ...");
        if (!BuildGidBuckets(linkage_list,
                             linkage_list_dir,
                             args.tmp_dir,
                             nbase,
                             args.gid_bucket_size,
                             args.progress_clusters,
                             &bucket_records_written,
                             &skipped_gid_out_of_range,
                             &err)) {
            LogError(err);
            return 1;
        }
        bucketize_sec = t.ElapsedSeconds();
    }

    if (!args.fast_dp && cfg.eval.linkage_preload_clusters_io_threads >= 0) {
        const Timer t;
        LogInfo("Phase2 prep: preloading all cluster data (io_threads=" +
                std::to_string(cfg.eval.linkage_preload_clusters_io_threads) + ") ...");
        if (!provider.PreloadAllClusters(cfg.eval.linkage_preload_clusters_io_threads, &preload_stats, &err)) {
            LogError(err.empty() ? "PreloadAllClusters failed." : err);
            return 1;
        }
        preload_sec = t.ElapsedSeconds();
        provider.ReleaseCoeffRawPayloads();
    }

    std::vector<eval::ClusterView> cv_by_cid;
    if (!args.fast_dp) {
        // Build stable per-cid views to avoid calling GetCluster in OMP parallel regions.
        cv_by_cid.resize(static_cast<std::size_t>(linkage_list.nlist()));
        const Timer t;
        for (int cid = 0; cid < linkage_list.nlist(); ++cid) {
            eval::ClusterView cv;
            eval::ClusterProvider::PrepStats prep{};
            if (!provider.GetCluster(cid, &cv, &prep, &err)) {
                LogError("GetCluster failed for cid=" + std::to_string(cid) + ": " + err);
                return 1;
            }
            cv_by_cid[static_cast<std::size_t>(cid)] = cv;
        }
        cluster_view_init_sec = t.ElapsedSeconds();
        provider.ReleaseCoeffRawPayloads();
    }

    const std::uint64_t nbucket = (nbase == 0) ? 0 : ((nbase + args.gid_bucket_size - 1) / args.gid_bucket_size);
    LogInfo("Phase2: scanning base sequentially by gid buckets (nbucket=" + std::to_string(nbucket) + ") ...");

    std::vector<std::uint32_t> cid_by_local;
    std::vector<std::uint32_t> pos_by_local;
    std::vector<float> rnorm2_by_local;
    std::vector<float> recon_by_local;
    std::vector<float> se_block;

    for (std::uint64_t b = 0; b < nbucket; ++b) {
        const std::uint64_t g0 = b * args.gid_bucket_size;
        const std::uint64_t bucket_len = std::min<std::uint64_t>(args.gid_bucket_size, nbase - g0);
        const std::string bucket_path = JoinPath(args.tmp_dir, "gid_bucket_" + std::to_string(b) + ".bin");
        std::uint64_t seen = 0;
        if (args.fast_dp) {
            const Timer t_load;
            if (!LoadReconBucketAny(args.tmp_dir,
                                    b,
                                    g0,
                                    bucket_len,
                                    C_root0.rows,
                                    args.strict_buckets != 0,
                                    &rnorm2_by_local,
                                    &recon_by_local,
                                    &seen,
                                    &err)) {
                LogError(err);
                return 1;
            }
            total_recon_bucket_load_sec += t_load.ElapsedSeconds();
        } else {
            const Timer t_load;
            if (!LoadBucket(bucket_path,
                            g0,
                            bucket_len,
                            args.strict_buckets != 0,
                            &cid_by_local,
                            &pos_by_local,
                            &seen,
                            &err)) {
                LogError(err);
                return 1;
            }
            total_bucket_load_sec += t_load.ElapsedSeconds();
        }
        if (seen < bucket_len) {
            missing_records += (bucket_len - seen);
        }

        std::uint64_t local_done = 0;
        while (local_done < bucket_len) {
            const int want = static_cast<int>(
                std::min<std::uint64_t>(static_cast<std::uint64_t>(args.block_cols), bucket_len - local_done));

            ColMajorMatrix<float> Xraw;
            {
                const Timer t_read;
                if (!stlq::io::ReadDatasetVectorBlockF32(base_r,
                                                         g0 + local_done,
                                                         static_cast<std::uint32_t>(want),
                                                         &Xraw,
                                                         &err)) {
                    LogError("ReadDatasetVectorBlockF32 failed: " + err);
                    return 1;
                }
                total_base_read_sec += t_read.ElapsedSeconds();
            }
            if (Xraw.rows != C_root0.rows || Xraw.cols != want) {
                LogError("Unexpected base block shape while scanning base.");
                return 1;
            }

            double rotate_sec = 0.0;
            if (!RotateBatchMaybeCuda(train.R,
                                      use_cuda,
                                      cuda_ptr,
                                      &Xraw,
                                      &rotate_sec,
                                      &err)) {
                LogError(err);
                return 1;
            }
            total_rotate_sec += rotate_sec;

            se_block.assign(static_cast<std::size_t>(want), std::numeric_limits<float>::quiet_NaN());
            const Timer t_recon;

            #pragma omp parallel
            {
                RunningStats local_stats;
                std::uint64_t local_written = 0;
                std::uint64_t local_fail = 0;
                std::uint64_t local_neg = 0;
                double local_min_raw = std::numeric_limits<double>::infinity();

                #pragma omp for schedule(static)
                for (int j = 0; j < want; ++j) {
                    const std::uint64_t local = local_done + static_cast<std::uint64_t>(j);
                    const float* x = Xraw.Col(j);

                    double xnorm = 0.0;
                    double dot = 0.0;
                    double rnorm2 = std::numeric_limits<double>::quiet_NaN();
                    if (args.fast_dp) {
                        const float rn_hint = rnorm2_by_local[static_cast<std::size_t>(local)];
                        if (!std::isfinite(rn_hint)) {
                            continue;
                        }
                        rnorm2 = 0.0;
                        const float* recon = recon_by_local.data() + static_cast<std::size_t>(local) * static_cast<std::size_t>(Xraw.rows);
                        // IMPORTANT:
                        // - recon is stored/loaded as fp16 (decoded to float).
                        // - To keep the SSE identity consistent (xnorm + rnorm2 - 2*dot >= 0),
                        //   compute rnorm2 from the decoded recon, instead of relying on a pre-stored
                        //   rnorm2 that may correspond to the pre-quantized (fp32) vector.
                        #pragma omp simd reduction(+:xnorm,dot,rnorm2)
                        for (int r = 0; r < Xraw.rows; ++r) {
                            const double xv = static_cast<double>(x[r]);
                            const double rv = static_cast<double>(recon[r]);
                            xnorm += xv * xv;
                            dot += xv * rv;
                            rnorm2 += rv * rv;
                        }
                    } else {
                        const std::uint32_t cid = cid_by_local[static_cast<std::size_t>(local)];
                        if (cid == kMissingCid) {
                            continue;
                        }
                        if (cid >= static_cast<std::uint32_t>(linkage_list.nlist())) {
                            ++local_fail;
                            continue;
                        }
                        const std::uint32_t pos = pos_by_local[static_cast<std::size_t>(local)];
                        const auto& cv = cv_by_cid[static_cast<std::size_t>(cid)];
                        if (pos >= static_cast<std::uint32_t>(std::max(0, cv.n_real))) {
                            ++local_fail;
                            continue;
                        }
                        #pragma omp simd reduction(+:xnorm)
                        for (int r = 0; r < Xraw.rows; ++r) {
                            const double xv = static_cast<double>(x[r]);
                            xnorm += xv * xv;
                        }

                        const float dot_f = ComputeSelfDotInt8(cv,
                                                               static_cast<int>(cid),
                                                               static_cast<int>(pos),
                                                               x,
                                                               Xraw.rows,
                                                               C_root0,
                                                               meta_root_small,
                                                               meta_one);
                        dot = static_cast<double>(dot_f);
                        const float rn = ReadSelfNorm2(cv, static_cast<int>(pos));
                        rnorm2 = static_cast<double>(rn);
                    }

                    const double sse_raw = xnorm + rnorm2 - 2.0 * dot;
                    // SSE must be non-negative; clamp small negative values caused by fp rounding.
                    const double sse = (sse_raw < 0.0) ? 0.0 : sse_raw;
                    if (sse_raw < 0.0) {
                        ++local_neg;
                        local_min_raw = std::min(local_min_raw, sse_raw);
                    }
                    const float se = static_cast<float>(sse);
                    se_block[static_cast<std::size_t>(j)] = se;
                    local_stats.Add(se);
                    if (std::isfinite(se)) {
                        ++local_written;
                    }
                }

                #pragma omp critical
                {
                    stats.Merge(local_stats);
                    written_real += local_written;
                    recon_failures += local_fail;
                    neg_se_count += local_neg;
                    min_se_raw = std::min(min_se_raw, local_min_raw);
                }
            }

            total_recon_sec += t_recon.ElapsedSeconds();

            if (write_se) {
                se_out.write(reinterpret_cast<const char*>(se_block.data()),
                             static_cast<std::streamsize>(static_cast<std::size_t>(want) * sizeof(float)));
                if (!se_out) {
                    LogError("Failed while writing se_by_gid at gid=" + std::to_string(g0 + local_done));
                    return 1;
                }
            }

            local_done += static_cast<std::uint64_t>(want);
        }

        const std::uint64_t done = std::min<std::uint64_t>((b + 1) * args.gid_bucket_size, nbase);
        LogInfo("Phase2 scanned " + std::to_string(done) + "/" + std::to_string(nbase));
    }

    if (write_se) {
        se_out.flush();
    }

    const double wall_sec = wall.ElapsedSeconds();
    const double mean_se = (stats.count > 0) ? (stats.sum / static_cast<double>(stats.count)) : 0.0;
    const double vec_per_sec = (wall_sec > 0.0) ? (static_cast<double>(nbase) / wall_sec) : 0.0;

    std::ofstream meta(std::filesystem::path(args.out_dir) / "stlq_int8_recon_mse_summary.txt",
                       std::ios::out | std::ios::trunc);
    if (!meta.is_open()) {
        LogError("Failed to open summary output file.");
        return 1;
    }
    meta << "run_root = " << run_root.string() << "\n";
    meta << "linkage_list_dir = " << linkage_list_dir.string() << "\n";
    meta << "se_file = " << (write_se ? se_path.string() : std::string("<none>")) << "\n";
    meta << "tmp_dir = " << args.tmp_dir << "\n";
    meta << "nbase_cfg = " << nbase_cfg << "\n";
    meta << "base_n = " << base_n << "\n";
    meta << "nbase = " << nbase << "\n";
    meta << "total_real = " << total_real << "\n";
    meta << "gid_bucket_size = " << args.gid_bucket_size << "\n";
    meta << "block_cols = " << args.block_cols << "\n";
    meta << "fast_dp = " << (args.fast_dp != 0 ? 1 : 0) << "\n";
    meta << "strict_buckets = " << (args.strict_buckets != 0 ? 1 : 0) << "\n";
    meta << "bucket_records_written = " << bucket_records_written << "\n";
    meta << "missing_records = " << missing_records << "\n";
    meta << "written_real = " << written_real << "\n";
    meta << "recon_failures = " << recon_failures << "\n";
    meta << "gid_out_of_range = " << skipped_gid_out_of_range << "\n";
    meta << "coverage = "
         << FormatDouble((nbase > 0) ? (static_cast<double>(written_real) / static_cast<double>(nbase)) : 0.0, 6)
         << "\n";
    meta << "mean_se = " << FormatDouble(mean_se, 8) << "\n";
    meta << "min_se = " << FormatDouble(std::isfinite(stats.min) ? stats.min : 0.0, 8) << "\n";
    meta << "max_se = " << FormatDouble(std::isfinite(stats.max) ? stats.max : 0.0, 8) << "\n";
    meta << "neg_se_count = " << neg_se_count << "\n";
    meta << "min_se_raw = " << FormatDouble(std::isfinite(min_se_raw) ? min_se_raw : 0.0, 8) << "\n";
    meta << "sse = " << FormatDouble(stats.sum, 8) << "\n";
    meta << "wall_sec = " << FormatDouble(wall_sec, 6) << "\n";
    meta << "bucketize_sec = " << FormatDouble(bucketize_sec, 6) << "\n";
    meta << "fast_dp_build_sec = " << FormatDouble(fast_dp_build_sec, 6) << "\n";
    meta << "preload_sec = " << FormatDouble(preload_sec, 6) << "\n";
    meta << "preload_coeff_cpu_decode_sec = " << FormatDouble(preload_stats.coeff_cpu_decode_sec, 6) << "\n";
    meta << "preload_louds_cpu_decode_sec = " << FormatDouble(preload_stats.louds_cpu_decode_sec, 6) << "\n";
    meta << "cluster_view_init_sec = " << FormatDouble(cluster_view_init_sec, 6) << "\n";
    meta << "base_read_sec = " << FormatDouble(total_base_read_sec, 6) << "\n";
    meta << "rotate_sec = " << FormatDouble(total_rotate_sec, 6) << "\n";
    meta << "recon_sec = " << FormatDouble(total_recon_sec, 6) << "\n";
    meta << "bucket_load_sec = " << FormatDouble(total_bucket_load_sec, 6) << "\n";
    meta << "recon_bucket_load_sec = " << FormatDouble(total_recon_bucket_load_sec, 6) << "\n";
    meta << "vectors_per_sec = " << FormatDouble(vec_per_sec, 2) << "\n";
    meta << "use_cuda_rotation = " << (use_cuda ? 1 : 0) << "\n";
    meta << "omp_threads = " << std::max(1, cfg.runtime.omp_threads) << "\n";
    meta.close();

    if (write_se && !keep_se) {
        std::error_code ec_rm;
        std::filesystem::remove(se_path, ec_rm);
        if (ec_rm) {
            LogInfo("[WARN] Failed to remove se_by_gid file: " + se_path.string());
        } else {
            LogInfo("Removed: " + se_path.string());
        }
    }

    if (!args.keep_tmp) {
        std::error_code ec3;
        std::filesystem::remove_all(args.tmp_dir, ec3);
    }

    if (write_se && keep_se) {
        LogInfo("Wrote: " + se_path.string());
    }
    LogInfo("Wrote: " + (std::filesystem::path(args.out_dir) / "stlq_int8_recon_mse_summary.txt").string());
    LogInfo("Int8 recon squared error mean=" + FormatDouble(mean_se, 8) +
            " vectors_per_sec=" + FormatDouble(vec_per_sec, 2));
    return 0;
}
