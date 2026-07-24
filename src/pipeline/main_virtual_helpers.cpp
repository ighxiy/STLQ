#include "stlq/pipeline/main_virtual_helpers.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>

#include <omp.h>

#include "stlq/pipeline/app_utils.h"
#include "stlq/core/blas.h"
#include "stlq/core/threading.h"
#include "stlq/common/logger.h"
#include "stlq/succinct/parent_louds.h"

namespace stlq::app {

// ---- Runtime helpers ----

std::string ZeroPad4(int v) {
    std::ostringstream oss;
    oss << std::setw(4) << std::setfill('0') << v;
    return oss.str();
}

std::filesystem::path NextRuntimeLogArchivePath(const std::filesystem::path& log_dir) {
    int max_seq = 0;
    std::error_code ec;
    if (std::filesystem::exists(log_dir, ec) && !ec) {
        for (const auto& entry : std::filesystem::directory_iterator(log_dir, ec)) {
            if (ec) break;
            if (!entry.is_regular_file()) continue;
            const std::string name = entry.path().filename().string();
            if (name.size() < 9) continue;
            if (name.rfind("run_", 0) != 0) continue;
            if (entry.path().extension() != ".log") continue;
            const std::string digits = entry.path().stem().string().substr(4);
            if (digits.empty()) continue;
            const bool all_digits = std::all_of(digits.begin(), digits.end(), [](unsigned char c) {
                return std::isdigit(c) != 0;
            });
            if (!all_digits) continue;
            try {
                max_seq = std::max(max_seq, std::stoi(digits));
            } catch (...) {
            }
        }
    }
    return log_dir / ("run_" + ZeroPad4(max_seq + 1) + ".log");
}

double RotateQueriesForEvalWithWarmup(const stlq::ColMajorMatrix<float>& R,
                                     stlq::ColMajorMatrix<float>* X) {
    if (!X || X->cols == 0 || stlq::IsIdentityRotation(R)) {
        return 0.0;
    }

    const int d_r = R.rows;
    const int wu_n = std::min(d_r, X->cols);
    std::vector<float> wu_out(static_cast<std::size_t>(d_r) * static_cast<std::size_t>(wu_n), 0.0f);
    stlq::ScopedBlasThreads blas_scope(stlq::OmpMaxThreads());
    stlq::GemmRaw(/*trans_a=*/false, /*trans_b=*/false,
                    /*m=*/d_r, /*n=*/wu_n, /*k=*/d_r,
                    /*alpha=*/1.0f,
                    /*A=*/R.data.data(), /*lda=*/d_r,
                    /*B=*/X->data.data(), /*ldb=*/X->rows,
                    /*beta=*/0.0f,
                    /*C=*/wu_out.data(), /*ldc=*/d_r);
    return stlq::ApplyRotationInPlaceGemmSeconds(R, X);
}

double Median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    if ((n % 2u) == 1u) {
        return v[n / 2u];
    }
    return 0.5 * (v[n / 2u - 1u] + v[n / 2u]);
}

stlq::Precomp BuildPrecompMetaOnly(const stlq::CodebookPack& pack) {
    stlq::Precomp pre;
    pre.d = pack.d;
    pre.m = static_cast<int>(pack.books.size());
    pre.h_vec = pack.h_vec;
    pre.offsets.assign(static_cast<std::size_t>(std::max(0, pre.m)), 0);
    int H = 0;
    for (int l = 0; l < pre.m; ++l) {
        pre.offsets[static_cast<std::size_t>(l)] = H;
        const int hl = (l < static_cast<int>(pre.h_vec.size())) ? pre.h_vec[static_cast<std::size_t>(l)] : 0;
        H += std::max(0, hl);
    }
    pre.H = H;
    pre.build_tag = pack.build_tag;
    return pre;
}

// ---- BaseBasic checkpoint helpers (resume/truncate) ----

namespace {

bool EnsureFileExists(const std::filesystem::path& p, std::string* err) {
    std::ofstream out(p, std::ios::binary | std::ios::app);
    if (!out.is_open()) {
        if (err) *err = "Failed to create file: " + p.string();
        return false;
    }
    return true;
}

}  // namespace

int ComputeNbucketsForBasic(const stlq::io::BaseBasicStoreConfig& store_cfg) {
    const int h0 = store_cfg.h_vec.empty() ? 0 : store_cfg.h_vec.front();
    const int bucket_size = std::max(1, store_cfg.bucket_size);
    return (h0 > 0) ? ((h0 + bucket_size - 1) / bucket_size) : 0;
}

std::filesystem::path BaseBasicCheckpointPath(const std::string& base_basic_dir) {
    return std::filesystem::path(base_basic_dir) / "checkpoint" / "base_basic" / "ckpt.bin";
}

bool ReadBaseBasicCheckpoint(const std::filesystem::path& path,
                            BaseBasicCheckpoint* out,
                            std::string* err) {
    if (!out) {
        if (err) *err = "ReadBaseBasicCheckpoint: out is null.";
        return false;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        if (err) *err = "ReadBaseBasicCheckpoint: failed to open: " + path.string();
        return false;
    }
    BaseBasicCheckpoint ck;
    in.read(reinterpret_cast<char*>(&ck.magic), sizeof(ck.magic));
    in.read(reinterpret_cast<char*>(&ck.version), sizeof(ck.version));
    in.read(reinterpret_cast<char*>(&ck.nbucket), sizeof(ck.nbucket));
    in.read(reinterpret_cast<char*>(&ck.reserved), sizeof(ck.reserved));
    in.read(reinterpret_cast<char*>(&ck.next_id), sizeof(ck.next_id));
    if (!in) {
        if (err) *err = "ReadBaseBasicCheckpoint: header read failed.";
        return false;
    }
    if (ck.magic != BaseBasicCheckpoint::kMagic || ck.version != 1) {
        if (err) *err = "ReadBaseBasicCheckpoint: bad magic/version.";
        return false;
    }
    ck.bucket_sizes.resize(static_cast<std::size_t>(ck.nbucket));
    if (ck.nbucket > 0) {
        in.read(reinterpret_cast<char*>(ck.bucket_sizes.data()),
                static_cast<std::streamsize>(ck.bucket_sizes.size() * sizeof(std::uint64_t)));
        if (!in) {
            if (err) *err = "ReadBaseBasicCheckpoint: bucket sizes read failed.";
            return false;
        }
    }
    *out = std::move(ck);
    return true;
}

bool WriteBaseBasicCheckpointAtomic(const std::filesystem::path& path,
                                   const BaseBasicCheckpoint& ck,
                                   std::string* err) {
    try {
        std::filesystem::create_directories(path.parent_path());
    } catch (...) {
        if (err) *err = "WriteBaseBasicCheckpoint: failed to create dir: " + path.parent_path().string();
        return false;
    }
    const std::filesystem::path tmp = path.parent_path() / (path.filename().string() + ".tmp");
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out.is_open()) {
            if (err) *err = "WriteBaseBasicCheckpoint: failed to open tmp: " + tmp.string();
            return false;
        }
        const std::uint32_t magic = BaseBasicCheckpoint::kMagic;
        const std::uint32_t version = 1;
        const std::uint32_t nbucket = ck.nbucket;
        const std::uint32_t reserved = 0;
        out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        out.write(reinterpret_cast<const char*>(&version), sizeof(version));
        out.write(reinterpret_cast<const char*>(&nbucket), sizeof(nbucket));
        out.write(reinterpret_cast<const char*>(&reserved), sizeof(reserved));
        out.write(reinterpret_cast<const char*>(&ck.next_id), sizeof(ck.next_id));
        if (nbucket > 0) {
            out.write(reinterpret_cast<const char*>(ck.bucket_sizes.data()),
                      static_cast<std::streamsize>(ck.bucket_sizes.size() * sizeof(std::uint64_t)));
        }
        if (!out) {
            if (err) *err = "WriteBaseBasicCheckpoint: write failed.";
            return false;
        }
    }
    try {
        std::filesystem::rename(tmp, path);
    } catch (...) {
        if (err) *err = "WriteBaseBasicCheckpoint: rename failed.";
        return false;
    }
    return true;
}

bool TruncateBaseBasicToCheckpoint(const std::string& out_dir,
                                  const stlq::io::BaseBasicStoreConfig& store_cfg,
                                  const BaseBasicCheckpoint& ck,
                                  std::string* err) {
    const std::filesystem::path dir(out_dir);
    const std::uint64_t next_id = ck.next_id;

    // cluster_id.u32
    {
        const auto p = dir / "cluster_id.u32";
        if (!EnsureFileExists(p, err)) return false;
        try {
            std::filesystem::resize_file(p, next_id * sizeof(std::uint32_t));
        } catch (...) {
            if (err) *err = "Failed to truncate: " + p.string();
            return false;
        }
    }

    // Current shard outputs
    const int shard_size = std::max(1, store_cfg.shard_size);
    const int m = std::max(1, store_cfg.m);
    const int m_codes = std::max(0, store_cfg.m - 1);
    const int shard = static_cast<int>(next_id / static_cast<std::uint64_t>(shard_size));
    const int in_shard = static_cast<int>(next_id % static_cast<std::uint64_t>(shard_size));
    {
        const std::string tag = ZeroPad4(shard);
        const auto codes_p = dir / ("codes_shard_" + tag + ".bin");
        const auto coeffs_p = dir / ("coeffs_shard_" + tag + ".bin");
        if (!EnsureFileExists(codes_p, err)) return false;
        if (!EnsureFileExists(coeffs_p, err)) return false;
        const std::uint64_t codes_bytes = static_cast<std::uint64_t>(in_shard) *
                                          static_cast<std::uint64_t>(m_codes) *
                                          static_cast<std::uint64_t>(sizeof(stlq::Code));
        const std::uint64_t coeffs_bytes = static_cast<std::uint64_t>(in_shard) *
                                           static_cast<std::uint64_t>(m) *
                                           static_cast<std::uint64_t>(sizeof(float));
        try {
            std::filesystem::resize_file(codes_p, codes_bytes);
            std::filesystem::resize_file(coeffs_p, coeffs_bytes);
        } catch (...) {
            if (err) *err = "Failed to truncate shard outputs.";
            return false;
        }
    }

    // Buckets
    const int nbucket = ComputeNbucketsForBasic(store_cfg);
    if (ck.nbucket != static_cast<std::uint32_t>(nbucket) ||
        ck.bucket_sizes.size() != static_cast<std::size_t>(nbucket)) {
        if (err) *err = "Checkpoint bucket layout mismatch.";
        return false;
    }
    for (int b = 0; b < nbucket; ++b) {
        const std::string tag = ZeroPad4(b);
        const auto p = dir / ("bucket_" + tag + ".bin");
        if (!EnsureFileExists(p, err)) return false;
        try {
            std::filesystem::resize_file(p, ck.bucket_sizes[static_cast<std::size_t>(b)]);
        } catch (...) {
            if (err) *err = "Failed to truncate bucket: " + p.string();
            return false;
        }
    }
    return true;
}

bool OnCommitBaseBasicCkpt(std::uint64_t next_id,
                          stlq::io::BaseBasicWriter* writer,
                          void* vctx,
                          std::string* err) {
    auto* ctx = static_cast<BaseBasicCkptCtx*>(vctx);
    if (!ctx) {
        if (err) *err = "OnCommitBaseBasicCkpt: ctx is null.";
        return false;
    }
    if (!writer) {
        if (err) *err = "OnCommitBaseBasicCkpt: writer is null.";
        return false;
    }
    BaseBasicCheckpoint ck;
    ck.nbucket = ctx->nbucket;
    ck.next_id = next_id;
    ck.bucket_sizes = writer->bucket_file_bytes();
    if (ck.bucket_sizes.size() != static_cast<std::size_t>(ck.nbucket)) {
        if (err) *err = "OnCommitBaseBasicCkpt: bucket sizes length mismatch.";
        return false;
    }
    return WriteBaseBasicCheckpointAtomic(ctx->path, ck, err);
}

// ---- Bit budget logging helpers (app-only diagnostics) ----

namespace {

inline std::uint64_t CeilDivU64(std::uint64_t a, std::uint64_t b) {
    return (a + b - 1) / b;
}

inline std::uint64_t SafeFileSizeBytes(const std::filesystem::path& p) {
    std::error_code ec;
    const bool ok = std::filesystem::exists(p, ec);
    if (!ok || ec) return 0ull;
    const auto sz = std::filesystem::file_size(p, ec);
    if (ec) return 0ull;
    return static_cast<std::uint64_t>(sz);
}

struct LinkageLOUDSBitBudget {
    std::uint64_t raw_bits_sum = 0;          // Σ(2n+1) over clusters (n includes virt)
    std::uint64_t header_bits_sum = 0;       // serialized header bits (meta)
    std::uint64_t padded_word_bits_sum = 0;  // padding bits to 64-bit words inside the blob
    std::uint64_t offsets_bits_disk = 0;     // parent_louds_offsets.u64 (meta)
    std::uint64_t louds_blob_bits_disk = 0;  // parent_louds.bin total bits (includes header + padding)
};

struct LinkageStoreBitBudgetStats {
    std::uint64_t n_real = 0;
    std::uint64_t n_virt = 0;
    std::uint64_t n_root_real = 0;
    double linkage_ratio = 0.0;  // (n_real - n_root_real) / n_real
};

static LinkageLOUDSBitBudget ComputeLinkageLOUDSBitBudget(const stlq::io::LinkageListReader& linkage_list) {
    LinkageLOUDSBitBudget b;
    if (!linkage_list.has_parent_louds()) {
        return b;
    }

    const std::filesystem::path dir(linkage_list.dir());
    b.louds_blob_bits_disk = SafeFileSizeBytes(dir / "parent_louds.bin") * 8ull;
    b.offsets_bits_disk = SafeFileSizeBytes(dir / "parent_louds_offsets.u64") * 8ull;

    const int nlist = linkage_list.nlist();
    std::uint64_t sum_bits_theory = 0;
    std::uint64_t sum_bits_words_padded = 0;

    for (int cid = 0; cid < nlist; ++cid) {
        std::uint64_t real_lo = 0, real_hi = 0, virt_lo = 0, virt_hi = 0, depth_lo = 0, depth_hi = 0;
        std::string ignored;
        if (!linkage_list.ReadClusterSpan(cid, &real_lo, &real_hi, &virt_lo, &virt_hi, &depth_lo, &depth_hi, &ignored)) {
            continue;
        }
        const std::uint64_t n_total = (real_hi - real_lo) + (virt_hi - virt_lo);
        const std::uint64_t bits_theory = 2ull * n_total + 1ull;
        sum_bits_theory += bits_theory;
        const std::uint64_t n_words = CeilDivU64(bits_theory, 64ull);
        sum_bits_words_padded += n_words * 64ull;
    }

    // louds_blob on disk stores: per-cluster Header(v1) + per-cluster LOUDS words (padded to 64-bit words).
    const std::uint64_t header_bytes_per_cluster =
        stlq::succinct::ParentLOUDS::SerializedBytesForNodes(0) - sizeof(std::uint64_t);  // n=0 => 1 word
    b.header_bits_sum = static_cast<std::uint64_t>(nlist) * header_bytes_per_cluster * 8ull;
    b.raw_bits_sum = sum_bits_theory;
    b.padded_word_bits_sum = (sum_bits_words_padded >= sum_bits_theory) ? (sum_bits_words_padded - sum_bits_theory) : 0ull;
    return b;
}

static LinkageStoreBitBudgetStats ComputeLinkageStoreBitBudgetStats(const stlq::io::LinkageListReader& linkage_list) {
    LinkageStoreBitBudgetStats s;
    s.n_real = linkage_list.total_real();
    s.n_virt = linkage_list.total_virtual();

    const int nlist = linkage_list.nlist();
    std::uint64_t sum_root_real = 0;
    std::vector<std::uint32_t> depth_offsets;
    for (int cid = 0; cid < nlist; ++cid) {
        std::uint32_t n_real_cluster = 0;
        std::string ignored;
        if (!linkage_list.ReadClusterDepthOffsets(cid, &depth_offsets, &n_real_cluster, &ignored)) {
            continue;
        }
        (void)n_real_cluster;
        if (!depth_offsets.empty()) {
            if (depth_offsets.size() >= 2) {
                sum_root_real += static_cast<std::uint64_t>(depth_offsets[1]);
            }
        }
    }
    s.n_root_real = sum_root_real;
    if (s.n_real > 0) {
        const std::uint64_t linkaged = (s.n_real > s.n_root_real) ? (s.n_real - s.n_root_real) : 0ull;
        s.linkage_ratio = static_cast<double>(linkaged) / static_cast<double>(s.n_real);
    }
    return s;
}

static LinkageLOUDSIndexBitBudget ComputeLinkageLOUDSIndexBitBudget(const stlq::Config& config,
                                                                const stlq::io::LinkageListReader& linkage_list) {
    LinkageLOUDSIndexBitBudget b;
    b.select_stride = static_cast<std::uint32_t>(std::max(1, config.eval.parent_louds_select_stride));
    b.rank_words_per_super_log2 = static_cast<std::uint32_t>(
        std::max(1, std::min(10, config.eval.parent_louds_rank_words_per_super_log2)));
    b.enabled = linkage_list.has_parent_louds() &&
                (config.eval.linkage_parent_louds_native_eval || config.eval.parent_louds_build_indices);
    if (!b.enabled) {
        return b;
    }

    const std::uint64_t total_real = linkage_list.total_real();
    const std::uint64_t words_per_super = 1ull << b.rank_words_per_super_log2;
    std::uint64_t rank_bits_sum = 0;
    std::uint64_t select_bits_sum = 0;
    const auto word_rank_bits = [&]() -> std::uint64_t {
        if (words_per_super <= 1ull) return 1ull;
        std::uint64_t x = 64ull * (words_per_super - 1ull);
        std::uint64_t bits = 0;
        while (x > 0ull) {
            ++bits;
            x >>= 1ull;
        }
        return std::max<std::uint64_t>(1ull, bits);
    }();

    for (int cid = 0; cid < linkage_list.nlist(); ++cid) {
        std::uint64_t real_lo = 0, real_hi = 0, virt_lo = 0, virt_hi = 0, depth_lo = 0, depth_hi = 0;
        std::string ignored;
        if (!linkage_list.ReadClusterSpan(cid, &real_lo, &real_hi, &virt_lo, &virt_hi, &depth_lo, &depth_hi, &ignored)) {
            continue;
        }
        const std::uint64_t n_total = (real_hi - real_lo) + (virt_hi - virt_lo);
        const std::uint64_t n_bits = 2ull * n_total + 1ull;
        const std::uint64_t n_words = CeilDivU64(n_bits, 64ull);
        const std::uint64_t n_supers = CeilDivU64(n_words, words_per_super);
        rank_bits_sum += (n_supers + 1ull) * 32ull + n_words * word_rank_bits;
        if (n_total > 0) {
            select_bits_sum += CeilDivU64(n_total, static_cast<std::uint64_t>(b.select_stride)) * 32ull;
        }
    }

    const std::uint64_t total_bits_sum = rank_bits_sum + select_bits_sum;
    if (total_real > 0) {
        const double denom = static_cast<double>(total_real);
        b.rank_bits_per_real = static_cast<double>(rank_bits_sum) / denom;
        b.select_bits_per_real = static_cast<double>(select_bits_sum) / denom;
        b.total_bits_per_real = static_cast<double>(total_bits_sum) / denom;
    }
    return b;
}

static std::string ToMiBString(std::uint64_t bytes) {
    const double mib = static_cast<double>(bytes) / (1024.0 * 1024.0);
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss << std::setprecision(3) << mib;
    return oss.str();
}

}  // namespace

void ApplyLOUDSIndexBudgetToTiming(const LinkageLOUDSIndexBitBudget& b,
                                  stlq::DiskLinkageEvalSession* session) {
    if (!session) return;
    session->louds_index_rank_bits_per_real = b.rank_bits_per_real;
    session->louds_index_select_bits_per_real = b.select_bits_per_real;
    session->louds_index_total_bits_per_real = b.total_bits_per_real;
    session->louds_index_effective_enabled = b.enabled ? 1 : 0;
    session->louds_index_select_stride = static_cast<int>(b.select_stride);
    session->louds_index_rank_words_per_super_log2 = static_cast<int>(b.rank_words_per_super_log2);
}

LinkageLOUDSIndexBitBudget LogLinkageBitBudgetSummary(const stlq::Config& config,
                                                 const stlq::io::LinkageListReader& linkage_list) {
    const LinkageStoreBitBudgetStats s = ComputeLinkageStoreBitBudgetStats(linkage_list);
    const LinkageLOUDSIndexBitBudget louds_index = ComputeLinkageLOUDSIndexBitBudget(config, linkage_list);
    if (s.n_real == 0) {
        return louds_index;
    }

    const int m = std::max(1, config.model.m);
    const int nlist = std::max(1, linkage_list.nlist());
    const double virt_ratio = static_cast<double>(s.n_virt) / static_cast<double>(s.n_real);
    const double root_ratio = static_cast<double>(s.n_root_real) / static_cast<double>(s.n_real);
    const double linkage_ratio = s.linkage_ratio;

    stlq::LogInfo("========== Linkage Bit Budget ==========");
    stlq::LogInfo("Linkage store stats: nreal=" + std::to_string(s.n_real) +
                    " nvirt=" + std::to_string(s.n_virt) +
                    " virt_ratio(nvirt/nreal)=" + std::to_string(virt_ratio) +
                    " nroot_real=" + std::to_string(s.n_root_real) +
                    " root_ratio=" + std::to_string(root_ratio) +
                    " linkage_ratio=" + std::to_string(linkage_ratio));

    const std::filesystem::path dir(linkage_list.dir());

    // LOUDS budget: keep header/offset/padding as metadata extra (not counted in theory budget).
    const LinkageLOUDSBitBudget louds = ComputeLinkageLOUDSBitBudget(linkage_list);
    stlq::LogInfo("Linkage parent LOUDS bit usage (accumulate by cluster): raw_bits=Σ(2n+1)=" +
                    std::to_string(louds.raw_bits_sum) +
                    " extra(header_bits=" + std::to_string(louds.header_bits_sum) +
                    " padded_word_bits=" + std::to_string(louds.padded_word_bits_sum) +
                    " offsets=" + std::to_string(louds.offsets_bits_disk) + ")");

    const double louds_raw_bits_per_real =
        (s.n_real > 0) ? (static_cast<double>(louds.raw_bits_sum) / static_cast<double>(s.n_real)) : 0.0;
    const double louds_disk_bits_per_real =
        (s.n_real > 0)
            ? (static_cast<double>(louds.louds_blob_bits_disk + louds.offsets_bits_disk) / static_cast<double>(s.n_real))
            : 0.0;

    stlq::LogInfo("Linkage parent LOUDS bits/real: louds_raw=" +
                    stlq::FormatFloat(static_cast<float>(louds_raw_bits_per_real), 6) +
                    " louds_disk(reference)=" +
                    stlq::FormatFloat(static_cast<float>(louds_disk_bits_per_real), 6));

    stlq::LogInfo("Linkage parent LOUDS index bits/real(just redundant dev, no used actually): rank=" +
                    stlq::FormatFloat(static_cast<float>(louds_index.rank_bits_per_real), 6) +
                    " select=" +
                    stlq::FormatFloat(static_cast<float>(louds_index.select_bits_per_real), 6) +
                    " total=" +
                    stlq::FormatFloat(static_cast<float>(louds_index.total_bits_per_real), 6));

    // Coeff codec budget: header is required for Huffman decoding => counted in coeff_all.
    const std::filesystem::path coeff_lens_path = dir / "coeff_lens.u8";
    const std::filesystem::path coeff_scales_path = dir / "coeff_scales.f32";
    const std::filesystem::path coeff_payload_path = dir / "coeff_payload.bin";
    const bool has_coeff_codec =
        std::filesystem::exists(coeff_lens_path) &&
        std::filesystem::exists(coeff_scales_path) &&
        std::filesystem::exists(coeff_payload_path);
    const std::uint64_t lens_bytes = has_coeff_codec ? SafeFileSizeBytes(coeff_lens_path) : 0;
    const std::uint64_t scales_bytes = has_coeff_codec ? SafeFileSizeBytes(coeff_scales_path) : 0;
    const std::uint64_t payload_bytes = has_coeff_codec ? SafeFileSizeBytes(coeff_payload_path) : 0;
    if (!has_coeff_codec) {
        stlq::LogInfo("Linkage coeff codec bytes: missing codec files in run dir; coeff codec bit budget skipped.");
    } else {
        stlq::LogInfo("Linkage coeff codec bytes: header(lens)=" +
                        std::to_string(lens_bytes) + " (" + ToMiBString(lens_bytes) + " MiB)" +
                        " header(scales)=" + std::to_string(scales_bytes) + " (" + ToMiBString(scales_bytes) + " MiB)" +
                        " data(payload)=" + std::to_string(payload_bytes) + " (" + ToMiBString(payload_bytes) + " MiB)");
    }

    const std::uint64_t ntotal = s.n_real + s.n_virt;
    const double denom_coeffs_nreal = static_cast<double>(s.n_real) * static_cast<double>(m);
    const double denom_coeffs_ntotal = static_cast<double>(ntotal) * static_cast<double>(m);

    const double coeff_data_bits_per_coeff_nreal =
        (denom_coeffs_nreal > 0.0) ? (8.0 * static_cast<double>(payload_bytes) / denom_coeffs_nreal) : 0.0;
    const double coeff_all_bits_per_coeff_nreal =
        (denom_coeffs_nreal > 0.0)
            ? (8.0 * (static_cast<double>(lens_bytes) + static_cast<double>(scales_bytes) + static_cast<double>(payload_bytes)) /
               denom_coeffs_nreal)
            : 0.0;
    const double coeff_data_bits_per_coeff_ntotal =
        (denom_coeffs_ntotal > 0.0) ? (8.0 * static_cast<double>(payload_bytes) / denom_coeffs_ntotal) : 0.0;
    const double coeff_all_bits_per_coeff_ntotal =
        (denom_coeffs_ntotal > 0.0)
            ? (8.0 * (static_cast<double>(lens_bytes) + static_cast<double>(scales_bytes) + static_cast<double>(payload_bytes)) /
               denom_coeffs_ntotal)
            : 0.0;

    if (has_coeff_codec) {
        stlq::LogInfo("Linkage coeff codec bits/real: coeff_data_nreal=" +
                        stlq::FormatFloat(static_cast<float>(coeff_data_bits_per_coeff_nreal), 6) +
                        " coeff_all_nreal(data+header)=" +
                        stlq::FormatFloat(static_cast<float>(coeff_all_bits_per_coeff_nreal), 6));

        stlq::LogInfo("Linkage coeff codec bits/ntotal: coeff_data_ntotal=" +
                        stlq::FormatFloat(static_cast<float>(coeff_data_bits_per_coeff_ntotal), 6) +
                        " coeff_all_ntotal(data+header)=" +
                        stlq::FormatFloat(static_cast<float>(coeff_all_bits_per_coeff_ntotal), 6));
    } else {
        stlq::LogInfo("Linkage coeff codec bits/real: skipped (coeff codec missing).");
        stlq::LogInfo("Linkage coeff codec bits/ntotal: skipped (coeff codec missing).");
    }

    // Summary #1: payload-only bits budget.
    const double payload_bits_per_vec =
        (1.0 - linkage_ratio) * (8.0 * static_cast<double>(m - 1)) +
        linkage_ratio * (8.0 * static_cast<double>(m)) +
        (coeff_all_bits_per_coeff_ntotal * static_cast<double>(m));

    stlq::LogInfo(
        "Summary of payload bits budget(root_code_real+linkage_code_real+coeff_all_ntotal): [(1-linkage_ratio) * 8 * (m-1)] + (linkage_ratio * 8 * m) + (coeff_all_ntotal * m)= " +
        stlq::FormatFloat(static_cast<float>(payload_bits_per_vec), 6) + " bit/per vector");

    // Summary #2: information bits budget with all structural overheads (includes louds_raw, virtual nodes, and IVF routing bits for all nodes).
    const double ivf_bits_theory = (nlist > 1) ? std::log2(static_cast<double>(nlist)) : 0.0;
    const double info_bits_per_vec =
        ivf_bits_theory + (1.0 - linkage_ratio + virt_ratio) * (8.0 * static_cast<double>(m - 1)) +
        linkage_ratio * (8.0 * static_cast<double>(m)) +
        (coeff_all_bits_per_coeff_nreal * static_cast<double>(m)) +
        louds_raw_bits_per_real;

    stlq::LogInfo(
        "Summary of information bits budget(ivf+root_code_virt+root_code_real+linkage_code_real+coeff_all_nreal+louds_raw): log2(nlist) + [(1-linkage_ratio+virt_ratio) * 8 * (m-1)] + (linkage_ratio * 8 * m) + (coeff_all_nreal * m) + louds_raw= " +
        stlq::FormatFloat(static_cast<float>(info_bits_per_vec), 6) + " bit/per vector");

    return louds_index;
}

// ---- Optional crash tracing on Windows (no-op on other platforms) ----

#if defined(_WIN32)

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <dbghelp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#pragma comment(lib, "dbghelp.lib")

namespace {

bool STLQCrashTraceEnabledImpl() {
    const char* v = std::getenv("STLQ_CRASH_TRACE");
    if (!v) return false;
    if (std::strcmp(v, "0") == 0) return false;
    if (_stricmp(v, "false") == 0) return false;
    return true;
}

void STLQPrintWinStackTraceImpl() {
    HANDLE proc = GetCurrentProcess();
    SymSetOptions(SYMOPT_DEFERRED_LOADS | SYMOPT_UNDNAME | SYMOPT_LOAD_LINES);
    if (!SymInitialize(proc, nullptr, TRUE)) {
        std::fprintf(stderr, "[LINKAGE] SymInitialize failed: gle=%lu\n", GetLastError());
        return;
    }

    void* frames[128];
    const USHORT n = CaptureStackBackTrace(/*FramesToSkip=*/0, /*FramesToCapture=*/128, frames, nullptr);
    std::fprintf(stderr, "[LINKAGE] Stack trace (%u frames):\n", static_cast<unsigned>(n));

    alignas(SYMBOL_INFO) unsigned char sym_buf[sizeof(SYMBOL_INFO) + 256];
    SYMBOL_INFO* sym = reinterpret_cast<SYMBOL_INFO*>(sym_buf);
    std::memset(sym_buf, 0, sizeof(sym_buf));
    sym->SizeOfStruct = sizeof(SYMBOL_INFO);
    sym->MaxNameLen = 255;

    for (USHORT i = 0; i < n; ++i) {
        DWORD64 addr = reinterpret_cast<DWORD64>(frames[i]);
        DWORD64 disp = 0;
        const BOOL ok = SymFromAddr(proc, addr, &disp, sym);

        IMAGEHLP_LINE64 line;
        std::memset(&line, 0, sizeof(line));
        line.SizeOfStruct = sizeof(line);
        DWORD disp32 = 0;
        const BOOL ok_line = SymGetLineFromAddr64(proc, addr, &disp32, &line);

        if (ok) {
            if (ok_line) {
                std::fprintf(stderr, "  #%02u %s + 0x%llx  (%s:%lu)\n",
                             static_cast<unsigned>(i),
                             sym->Name,
                             static_cast<unsigned long long>(disp),
                             line.FileName,
                             static_cast<unsigned long>(line.LineNumber));
            } else {
                std::fprintf(stderr, "  #%02u %s + 0x%llx\n",
                             static_cast<unsigned>(i),
                             sym->Name,
                             static_cast<unsigned long long>(disp));
            }
        } else {
            std::fprintf(stderr, "  #%02u 0x%llx\n",
                         static_cast<unsigned>(i),
                         static_cast<unsigned long long>(addr));
        }
    }

    SymCleanup(proc);
}

LONG WINAPI STLQUnhandledExceptionFilterImpl(EXCEPTION_POINTERS* ep) {
    if (ep && ep->ExceptionRecord) {
        const DWORD code = ep->ExceptionRecord->ExceptionCode;
        const void* addr = ep->ExceptionRecord->ExceptionAddress;
        std::fprintf(stderr, "[LINKAGE] Unhandled exception: code=0x%08lx addr=%p\n",
                     static_cast<unsigned long>(code), addr);
    } else {
        std::fprintf(stderr, "[LINKAGE] Unhandled exception (no info)\n");
    }
    STLQPrintWinStackTraceImpl();
    std::fflush(stderr);
    return EXCEPTION_EXECUTE_HANDLER;
}

}  // namespace

void InstallCrashTraceIfEnabled() {
    if (STLQCrashTraceEnabledImpl()) {
        SetUnhandledExceptionFilter(STLQUnhandledExceptionFilterImpl);
    }
}

#else

void InstallCrashTraceIfEnabled() {
}

#endif

}  // namespace stlq::app
