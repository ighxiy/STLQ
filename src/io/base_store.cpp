#include "stlq/io/base_store.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <cstdint>
#include <sstream>

namespace stlq::io {

namespace {

std::string JoinPath(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    if (a.back() == '/' || a.back() == '\\') return a + b;
    return a + "/" + b;
}

std::string ZeroPad(int value, int width) {
    std::ostringstream oss;
    oss << std::setw(width) << std::setfill('0') << value;
    return oss.str();
}

bool EnsureDir(const std::string& dir, std::string* err) {
    try {
        std::filesystem::create_directories(dir);
        return true;
    } catch (...) {
        if (err) {
            *err = "BaseBasicWriter: failed to create dir: " + dir;
        }
        return false;
    }
}

bool WriteMetaBin(const std::string& path, const BaseBasicStoreConfig& cfg, std::string* err) {
    // Binary, little-endian. Versioned for stable parsing in out-of-core pipelines.
    struct Header {
        std::uint32_t magic = 0x53414243u;   // "CBAS"
        std::uint32_t version = 2;
        std::int32_t d = 0;
        std::int32_t m = 0;
        std::int32_t m_codes = 0;
        std::int32_t shard_size = 0;
        std::int32_t bucket_size = 0;
        std::uint32_t flags = 0;
        std::int32_t h_len = 0;
        std::int32_t raw_bytes_per_vec = 0;
    } h;
    h.d = cfg.d;
    h.m = cfg.m;
    h.m_codes = std::max(0, cfg.m - 1);
    h.shard_size = cfg.shard_size;
    h.bucket_size = cfg.bucket_size;
    h.flags = (cfg.write_vector_bucket ? 1u : 0u) | (cfg.write_basic_to_bucket ? 2u : 0u);
    h.h_len = static_cast<std::int32_t>(cfg.h_vec.size());
    h.raw_bytes_per_vec = cfg.write_vector_bucket ? std::max(0, cfg.raw_bytes_per_vec) : 0;
    if (cfg.write_vector_bucket && h.raw_bytes_per_vec == 0) {
        // Backward-compatible default: treat raw vectors as u8 (d bytes per vector).
        h.raw_bytes_per_vec = cfg.d;
    }

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        if (err) {
            *err = "BaseBasicWriter: failed to open meta.bin.";
        }
        return false;
    }
    out.write(reinterpret_cast<const char*>(&h), sizeof(Header));
    if (!cfg.h_vec.empty()) {
        out.write(reinterpret_cast<const char*>(cfg.h_vec.data()),
                  static_cast<std::streamsize>(cfg.h_vec.size() * sizeof(int)));
    }
    if (!out) {
        if (err) {
            *err = "BaseBasicWriter: failed to write meta.bin.";
        }
        return false;
    }
    return true;
}

}  // namespace

bool BaseBasicWriter::Open(const BaseBasicStoreConfig& cfg, std::string* err) {
    cfg_ = cfg;
    current_shard_ = -1;
    current_shard_begin_ = 0;
    resume_mode_ = false;
    resume_start_id_ = 0;
    resume_shard_ = 0;
    resume_in_shard_ = 0;
    buckets_.clear();
    bucket_file_bytes_.clear();
    bucket_flush_bytes_ = static_cast<std::size_t>(std::max(1, cfg_.bucket_flush_mb)) * 1024ull * 1024ull;

    if (cfg_.dir.empty() || cfg_.d <= 0 || cfg_.m <= 0) {
        if (err) {
            *err = "BaseBasicWriter::Open: invalid config.";
        }
        return false;
    }
    if (cfg_.write_vector_bucket) {
        if (cfg_.raw_bytes_per_vec <= 0) {
            cfg_.raw_bytes_per_vec = cfg_.d;
        }
    } else {
        cfg_.raw_bytes_per_vec = 0;
    }
    if (!EnsureDir(cfg_.dir, err)) {
        return false;
    }

    // Important: make reruns with the same (output_dir, dataset, pre_fix) deterministic.
    // Avoid directory scans (can be very slow for large output dirs). Instead, deterministically
    // truncate all possible bucket files for this run so no old data can leak in even if some
    // buckets end up empty (never opened lazily).
    const int h0 = cfg_.h_vec.empty() ? 0 : cfg_.h_vec.front();
    const int bucket_size_for_cleanup = std::max(1, cfg_.bucket_size);
    const int nbucket = (h0 > 0) ? ((h0 + bucket_size_for_cleanup - 1) / bucket_size_for_cleanup) : 0;
    if (nbucket > 0) {
        buckets_.resize(static_cast<std::size_t>(nbucket));
        bucket_file_bytes_.assign(static_cast<std::size_t>(nbucket), 0ull);
        for (int b = 0; b < nbucket; ++b) {
            const std::string bucket_tag = ZeroPad(b, 4);
            const std::string path = JoinPath(cfg_.dir, "bucket_" + bucket_tag + ".bin");
            std::ofstream tmp(path, std::ios::binary | std::ios::trunc);
            (void)tmp;
        }
    }

    // Meta (minimal json; extended later).
    {
        const std::string meta_path = JoinPath(cfg_.dir, "meta.json");
        std::ofstream meta(meta_path, std::ios::binary);
        if (!meta.is_open()) {
            if (err) {
                *err = "BaseBasicWriter::Open: failed to open meta.json.";
            }
            return false;
        }
        meta << "{\n";
        meta << "  \"d\": " << cfg_.d << ",\n";
        meta << "  \"m\": " << cfg_.m << ",\n";
        meta << "  \"m_codes\": " << std::max(0, cfg_.m - 1) << ",\n";
        meta << "  \"shard_size\": " << cfg_.shard_size << ",\n";
        meta << "  \"bucket_size\": " << cfg_.bucket_size << ",\n";
        meta << "  \"write_vector_bucket\": " << (cfg_.write_vector_bucket ? "true" : "false") << ",\n";
        meta << "  \"write_basic_to_bucket\": " << (cfg_.write_basic_to_bucket ? "true" : "false") << ",\n";
        meta << "  \"raw_bytes_per_vec\": " << cfg_.raw_bytes_per_vec << ",\n";
        meta << "  \"h_vec\": [";
        for (std::size_t i = 0; i < cfg_.h_vec.size(); ++i) {
            if (i) meta << ", ";
            meta << cfg_.h_vec[i];
        }
        meta << "]\n";
        meta << "}\n";
    }

    // Binary meta for robust parsing (preferred over meta.json).
    if (!WriteMetaBin(JoinPath(cfg_.dir, "meta.bin"), cfg_, err)) {
        return false;
    }

    const std::string cluster_id_path = JoinPath(cfg_.dir, "cluster_id.u32");
    cluster_id_out_.open(cluster_id_path, std::ios::binary | std::ios::trunc);
    if (!cluster_id_out_.is_open()) {
        if (err) {
            *err = "BaseBasicWriter::Open: failed to open cluster_id.u32.";
        }
        return false;
    }

    if (cfg_.bucket_size <= 0) {
        cfg_.bucket_size = 256;
    }
    // Buckets are created lazily on first write; size will be fixed later when we know h0.
    return true;
}

bool BaseBasicWriter::OpenResume(const BaseBasicStoreConfig& cfg,
                                std::uint64_t start_id,
                                const std::vector<std::uint64_t>& bucket_sizes,
                                std::string* err) {
    cfg_ = cfg;
    current_shard_ = -1;
    current_shard_begin_ = 0;
    resume_mode_ = true;
    resume_start_id_ = start_id;
    buckets_.clear();
    bucket_file_bytes_.clear();
    bucket_flush_bytes_ = static_cast<std::size_t>(std::max(1, cfg_.bucket_flush_mb)) * 1024ull * 1024ull;

    if (cfg_.dir.empty() || cfg_.d <= 0 || cfg_.m <= 0) {
        if (err) {
            *err = "BaseBasicWriter::OpenResume: invalid config.";
        }
        return false;
    }
    if (cfg_.write_vector_bucket) {
        if (cfg_.raw_bytes_per_vec <= 0) {
            cfg_.raw_bytes_per_vec = cfg_.d;
        }
    } else {
        cfg_.raw_bytes_per_vec = 0;
    }
    if (cfg_.bucket_size <= 0) {
        cfg_.bucket_size = 256;
    }
    if (!EnsureDir(cfg_.dir, err)) {
        return false;
    }

    const int shard_size = std::max(1, cfg_.shard_size);
    resume_shard_ = static_cast<int>(start_id / static_cast<std::uint64_t>(shard_size));
    resume_in_shard_ = static_cast<int>(start_id % static_cast<std::uint64_t>(shard_size));

    const int h0 = cfg_.h_vec.empty() ? 0 : cfg_.h_vec.front();
    const int bucket_size_for_layout = std::max(1, cfg_.bucket_size);
    const int nbucket = (h0 > 0) ? ((h0 + bucket_size_for_layout - 1) / bucket_size_for_layout) : 0;
    if (nbucket > 0) {
        buckets_.resize(static_cast<std::size_t>(nbucket));
    }

    if (static_cast<int>(bucket_sizes.size()) != nbucket) {
        if (err) {
            *err = "BaseBasicWriter::OpenResume: bucket_sizes length mismatch (got=" +
                   std::to_string(bucket_sizes.size()) + ", expected=" + std::to_string(nbucket) + ").";
        }
        return false;
    }
    bucket_file_bytes_ = bucket_sizes;

    const std::string cluster_id_u32 = JoinPath(cfg_.dir, "cluster_id.u32");
    std::error_code ec;
    if (!std::filesystem::exists(cluster_id_u32, ec)) {
        if (err) {
            *err = "BaseBasicWriter::OpenResume: missing cluster_id.u32 under: " + cfg_.dir;
        }
        return false;
    }
    cluster_id_out_.open(cluster_id_u32, std::ios::binary | std::ios::app);
    if (!cluster_id_out_.is_open()) {
        if (err) {
            *err = "BaseBasicWriter::OpenResume: failed to open cluster_id.u32: " + cluster_id_u32;
        }
        return false;
    }
    return true;
}

// NOTE: BaseBasicWriter currently has no explicit Close() in the public API; make sure we flush
// any pending bucket buffers when the writer is destroyed (best-effort, no error reporting).
BaseBasicWriter::~BaseBasicWriter() {
    std::string ignored;
    (void)Close(&ignored);
}

bool BaseBasicWriter::Close(std::string* err) {
    for (std::size_t i = 0; i < buckets_.size(); ++i) {
        if (!FlushBucket(static_cast<int>(i), err)) {
            return false;
        }
        if (buckets_[i].open && buckets_[i].out.is_open()) {
            buckets_[i].out.close();
        }
        buckets_[i].open = false;
        buckets_[i].buf.clear();
    }
    if (codes_out_.is_open()) codes_out_.close();
    if (coeffs_out_.is_open()) coeffs_out_.close();
    if (cluster_id_out_.is_open()) {
        cluster_id_out_.flush();
        cluster_id_out_.close();
    }
    current_shard_ = -1;
    return true;
}

bool BaseBasicWriter::Flush(std::string* err) {
    for (std::size_t i = 0; i < buckets_.size(); ++i) {
        if (!FlushBucket(static_cast<int>(i), err)) {
            return false;
        }
        if (buckets_[i].open && buckets_[i].out.is_open()) {
            buckets_[i].out.flush();
            if (!buckets_[i].out) {
                if (err) *err = "BaseBasicWriter::Flush: bucket flush failed.";
                return false;
            }
        }
    }
    if (codes_out_.is_open()) {
        codes_out_.flush();
        if (!codes_out_) {
            if (err) *err = "BaseBasicWriter::Flush: codes shard flush failed.";
            return false;
        }
    }
    if (coeffs_out_.is_open()) {
        coeffs_out_.flush();
        if (!coeffs_out_) {
            if (err) *err = "BaseBasicWriter::Flush: coeffs shard flush failed.";
            return false;
        }
    }
    if (cluster_id_out_.is_open()) {
        cluster_id_out_.flush();
        if (!cluster_id_out_) {
            if (err) *err = "BaseBasicWriter::Flush: cluster_id flush failed.";
            return false;
        }
    }
    return true;
}

bool BaseBasicWriter::EnsureShard(std::uint64_t global_id, std::string* err) {
    const int shard_size = std::max(1, cfg_.shard_size);
    const int shard = static_cast<int>(global_id / static_cast<std::uint64_t>(shard_size));
    if (shard == current_shard_) {
        return true;
    }

    if (resume_mode_ && global_id < resume_start_id_) {
        if (err) {
            *err = "BaseBasicWriter::EnsureShard(resume): global_id < resume_start_id.";
        }
        return false;
    }

    current_shard_ = shard;
    current_shard_begin_ = static_cast<std::uint64_t>(shard) * static_cast<std::uint64_t>(shard_size);

    if (codes_out_.is_open()) codes_out_.close();
    if (coeffs_out_.is_open()) coeffs_out_.close();

    const std::string shard_tag = ZeroPad(shard, 4);
    const std::string codes_path = JoinPath(cfg_.dir, "codes_shard_" + shard_tag + ".bin");
    const std::string coeffs_path = JoinPath(cfg_.dir, "coeffs_shard_" + shard_tag + ".bin");
    if (resume_mode_) {
        const bool append = (shard == resume_shard_ && resume_in_shard_ > 0);
        const std::ios::openmode mode = std::ios::binary | (append ? std::ios::app : std::ios::trunc);
        codes_out_.open(codes_path, mode);
        coeffs_out_.open(coeffs_path, mode);
    } else {
        codes_out_.open(codes_path, std::ios::binary | std::ios::trunc);
        coeffs_out_.open(coeffs_path, std::ios::binary | std::ios::trunc);
    }
    if (!codes_out_.is_open() || !coeffs_out_.is_open()) {
        if (err) {
            *err = "BaseBasicWriter: failed to open shard outputs.";
        }
        return false;
    }
    return true;
}

bool BaseBasicWriter::EnsureBucket(int bucket_id, std::string* err) {
    if (bucket_id < 0) {
        if (err) *err = "BaseBasicWriter: invalid bucket id.";
        return false;
    }
    if (bucket_id >= static_cast<int>(buckets_.size())) {
        buckets_.resize(static_cast<std::size_t>(bucket_id + 1));
    }
    auto& b = buckets_[static_cast<std::size_t>(bucket_id)];
    if (b.open) {
        return true;
    }
    const std::string bucket_tag = ZeroPad(bucket_id, 4);
    const std::string path = JoinPath(cfg_.dir, "bucket_" + bucket_tag + ".bin");
    // Use trunc for fresh runs to avoid mixing old outputs. For resume, append at the checkpoint-truncated tail.
    b.out.open(path, std::ios::binary | (resume_mode_ ? std::ios::app : std::ios::trunc));
    if (!b.out.is_open()) {
        if (err) {
            *err = "BaseBasicWriter: failed to open bucket file: " + path;
        }
        return false;
    }
    b.open = true;
    b.buf.clear();
    b.buf.reserve(std::min<std::size_t>(bucket_flush_bytes_, 8ull * 1024ull * 1024ull));
    return true;
}

bool BaseBasicWriter::FlushBucket(int bucket_id, std::string* err) {
    if (bucket_id < 0 || bucket_id >= static_cast<int>(buckets_.size())) {
        return true;
    }
    auto& b = buckets_[static_cast<std::size_t>(bucket_id)];
    if (!b.open || b.buf.empty()) {
        return true;
    }
    b.out.write(reinterpret_cast<const char*>(b.buf.data()),
                static_cast<std::streamsize>(b.buf.size()));
    if (!b.out) {
        if (err) {
            *err = "BaseBasicWriter: bucket write failed.";
        }
        return false;
    }
    if (bucket_id >= 0 && bucket_id < static_cast<int>(bucket_file_bytes_.size())) {
        bucket_file_bytes_[static_cast<std::size_t>(bucket_id)] += static_cast<std::uint64_t>(b.buf.size());
    }
    b.buf.clear();
    return true;
}

bool BaseBasicWriter::AppendBlock(std::uint64_t start_id,
                                 const std::vector<std::uint32_t>& cluster_id,
                                 const ColMajorMatrix<Code>& B_small,
                                 const ColMajorMatrix<float>& a,
                                 const ColMajorMatrix<std::uint8_t>* x_u8_optional,
                                 std::string* err) {
    const int nblk = static_cast<int>(cluster_id.size());
    if (nblk <= 0) {
        return true;
    }
    const int m_codes = std::max(0, cfg_.m - 1);
    if (B_small.rows != m_codes || B_small.cols != nblk || a.rows != cfg_.m || a.cols != nblk) {
        if (err) {
            *err = "BaseBasicWriter::AppendBlock: B/a shape mismatch.";
        }
        return false;
    }
    if (x_u8_optional) {
        if (x_u8_optional->rows != cfg_.d || x_u8_optional->cols != nblk) {
            if (err) {
                *err = "BaseBasicWriter::AppendBlock: x_u8 shape mismatch.";
            }
            return false;
        }
    }

    // cluster_id.u32 (global order).
    cluster_id_out_.write(reinterpret_cast<const char*>(cluster_id.data()),
                          static_cast<std::streamsize>(cluster_id.size() * sizeof(std::uint32_t)));
    if (!cluster_id_out_) {
        if (err) {
            *err = "BaseBasicWriter::AppendBlock: failed to write cluster_id file.";
        }
        return false;
    }

    // codes/coeffs shards (global order).
    if (!EnsureShard(start_id, err)) {
        return false;
    }
    const std::size_t n_codes =
        static_cast<std::size_t>(B_small.rows) * static_cast<std::size_t>(nblk);
    const std::size_t n_coeffs =
        static_cast<std::size_t>(a.rows) * static_cast<std::size_t>(nblk);
    codes_out_.write(reinterpret_cast<const char*>(B_small.data.data()),
                     static_cast<std::streamsize>(n_codes * sizeof(Code)));
    coeffs_out_.write(reinterpret_cast<const char*>(a.data.data()),
                      static_cast<std::streamsize>(n_coeffs * sizeof(float)));
    if (!codes_out_ || !coeffs_out_) {
        if (err) {
            *err = "BaseBasicWriter::AppendBlock: failed to write shard outputs.";
        }
        return false;
    }

    // Optional bucket record (for fast cluster-local downstream processing).
    if (!cfg_.write_vector_bucket && !cfg_.write_basic_to_bucket) {
        return true;
    }

    if (cfg_.write_vector_bucket && !x_u8_optional) {
        if (err) {
            *err = "BaseBasicWriter::AppendBlock: write_vector_bucket=true but x_u8_optional is null.";
        }
        return false;
    }

    const int d = cfg_.d;
    const int raw_bytes_per_vec = cfg_.write_vector_bucket ? cfg_.raw_bytes_per_vec : 0;
    if (cfg_.write_vector_bucket && raw_bytes_per_vec != d) {
        if (err) {
            *err = "BaseBasicWriter::AppendBlock: raw_bytes_per_vec != d; use AppendBlockRawF32 for float datasets.";
        }
        return false;
    }
    const int m = cfg_.m;
    const int m_codes_bucket = m_codes;
    const int bucket_size = std::max(1, cfg_.bucket_size);
    const std::size_t rec_bytes =
        sizeof(std::uint32_t) + sizeof(std::uint32_t) +
        (cfg_.write_vector_bucket ? static_cast<std::size_t>(raw_bytes_per_vec) : 0ull) +
        (cfg_.write_basic_to_bucket ? (static_cast<std::size_t>(m_codes_bucket) +
                                       static_cast<std::size_t>(m) * sizeof(float))
                                    : 0ull);

    for (int i = 0; i < nblk; ++i) {
        const std::uint64_t gid = start_id + static_cast<std::uint64_t>(i);
        const std::uint32_t cid = cluster_id[static_cast<std::size_t>(i)];
        const int bucket_id = static_cast<int>(cid / static_cast<std::uint32_t>(bucket_size));
        if (!EnsureBucket(bucket_id, err)) {
            return false;
        }
        auto& b = buckets_[static_cast<std::size_t>(bucket_id)];

        // record layout:
        // - uint32 global_id
        // - uint32 cluster_id
        // - optional uint8 x[d]
        // - optional Code B[m_codes]
        // - optional float a[m]
        //
        // Performance note: avoid `vector<uint8_t>::resize()` per record because it value-initializes
        // (zero-fills) the new bytes; instead append via `insert()` which only copies our payload.
        // Ensure capacity without calling reserve() every record (which can cause frequent reallocations).
        if (b.buf.capacity() - b.buf.size() < rec_bytes) {
            const std::size_t need = b.buf.size() + rec_bytes;
            const std::size_t grow =
                (b.buf.capacity() > 0) ? (b.buf.capacity() + (b.buf.capacity() >> 1)) : (8ull * 1024ull * 1024ull);
            b.buf.reserve(std::max(need, grow));
        }
        auto append_bytes = [&](const void* ptr, std::size_t bytes) {
            const auto* p = static_cast<const std::uint8_t*>(ptr);
            b.buf.insert(b.buf.end(), p, p + bytes);
        };

        const auto gid32 = static_cast<std::uint32_t>(gid);
        append_bytes(&gid32, sizeof(std::uint32_t));
        append_bytes(&cid, sizeof(std::uint32_t));

        if (cfg_.write_vector_bucket) {
            const std::uint8_t* x = x_u8_optional->Col(i);
            append_bytes(x, static_cast<std::size_t>(raw_bytes_per_vec));
        }
        if (cfg_.write_basic_to_bucket) {
            const Code* bcol = B_small.Col(i);
            append_bytes(bcol, static_cast<std::size_t>(m_codes_bucket));
            const float* acol = a.Col(i);
            append_bytes(acol, static_cast<std::size_t>(m) * sizeof(float));
        }

        if (b.buf.size() >= bucket_flush_bytes_) {
            if (!FlushBucket(bucket_id, err)) {
                return false;
            }
        }
    }
    return true;
}

bool BaseBasicWriter::AppendBlockRawU8(std::uint64_t start_id,
                                      const std::vector<std::uint32_t>& cluster_id,
                                      const ColMajorMatrix<Code>& B_small,
                                      const ColMajorMatrix<float>& a,
                                      const std::uint8_t* x_u8,
                                      int ld_x_u8,
                                      std::string* err) {
    const int nblk = static_cast<int>(cluster_id.size());
    if (nblk <= 0) {
        return true;
    }
    const int m_codes = std::max(0, cfg_.m - 1);
    if (B_small.rows != m_codes || B_small.cols != nblk || a.rows != cfg_.m || a.cols != nblk) {
        if (err) {
            *err = "BaseBasicWriter::AppendBlockRawU8: B/a shape mismatch.";
        }
        return false;
    }
    if (cfg_.write_vector_bucket) {
        if (!x_u8) {
            if (err) {
                *err = "BaseBasicWriter::AppendBlockRawU8: write_vector_bucket=true but x_u8 is null.";
            }
            return false;
        }
        if (ld_x_u8 < cfg_.d) {
            if (err) {
                *err = "BaseBasicWriter::AppendBlockRawU8: invalid ld_x_u8.";
            }
            return false;
        }
        if (cfg_.raw_bytes_per_vec != cfg_.d) {
            if (err) {
                *err = "BaseBasicWriter::AppendBlockRawU8: raw_bytes_per_vec != d; use AppendBlockRawF32 for float datasets.";
            }
            return false;
        }
    }

    // cluster_id.u32 (global order).
    cluster_id_out_.write(reinterpret_cast<const char*>(cluster_id.data()),
                          static_cast<std::streamsize>(cluster_id.size() * sizeof(std::uint32_t)));
    if (!cluster_id_out_) {
        if (err) {
            *err = "BaseBasicWriter::AppendBlockRawU8: failed to write cluster_id file.";
        }
        return false;
    }

    // codes/coeffs shards (global order).
    if (!EnsureShard(start_id, err)) {
        return false;
    }
    const std::size_t n_codes =
        static_cast<std::size_t>(B_small.rows) * static_cast<std::size_t>(nblk);
    const std::size_t n_coeffs =
        static_cast<std::size_t>(a.rows) * static_cast<std::size_t>(nblk);
    codes_out_.write(reinterpret_cast<const char*>(B_small.data.data()),
                     static_cast<std::streamsize>(n_codes * sizeof(Code)));
    coeffs_out_.write(reinterpret_cast<const char*>(a.data.data()),
                      static_cast<std::streamsize>(n_coeffs * sizeof(float)));
    if (!codes_out_ || !coeffs_out_) {
        if (err) {
            *err = "BaseBasicWriter::AppendBlockRawU8: failed to write shard outputs.";
        }
        return false;
    }

    // Optional bucket record (for fast cluster-local downstream processing).
    if (!cfg_.write_vector_bucket && !cfg_.write_basic_to_bucket) {
        return true;
    }

    const int raw_bytes_per_vec = cfg_.write_vector_bucket ? cfg_.raw_bytes_per_vec : 0;
    const int m = cfg_.m;
    const int m_codes_bucket = m_codes;
    const int bucket_size = std::max(1, cfg_.bucket_size);
    const std::size_t rec_bytes =
        sizeof(std::uint32_t) + sizeof(std::uint32_t) +
        (cfg_.write_vector_bucket ? static_cast<std::size_t>(raw_bytes_per_vec) : 0ull) +
        (cfg_.write_basic_to_bucket ? (static_cast<std::size_t>(m_codes_bucket) +
                                       static_cast<std::size_t>(m) * sizeof(float))
                                    : 0ull);

    for (int i = 0; i < nblk; ++i) {
        const std::uint64_t gid = start_id + static_cast<std::uint64_t>(i);
        const std::uint32_t cid = cluster_id[static_cast<std::size_t>(i)];
        const int bucket_id = static_cast<int>(cid / static_cast<std::uint32_t>(bucket_size));
        if (!EnsureBucket(bucket_id, err)) {
            return false;
        }
        auto& b = buckets_[static_cast<std::size_t>(bucket_id)];

        if (b.buf.capacity() - b.buf.size() < rec_bytes) {
            const std::size_t need = b.buf.size() + rec_bytes;
            const std::size_t grow =
                (b.buf.capacity() > 0) ? (b.buf.capacity() + (b.buf.capacity() >> 1)) : (8ull * 1024ull * 1024ull);
            b.buf.reserve(std::max(need, grow));
        }
        auto append_bytes = [&](const void* ptr, std::size_t bytes) {
            const auto* p = static_cast<const std::uint8_t*>(ptr);
            b.buf.insert(b.buf.end(), p, p + bytes);
        };

        const auto gid32 = static_cast<std::uint32_t>(gid);
        append_bytes(&gid32, sizeof(std::uint32_t));
        append_bytes(&cid, sizeof(std::uint32_t));

        if (cfg_.write_vector_bucket) {
            const std::uint8_t* x = x_u8 + static_cast<std::size_t>(i) * static_cast<std::size_t>(ld_x_u8);
            append_bytes(x, static_cast<std::size_t>(raw_bytes_per_vec));
        }
        if (cfg_.write_basic_to_bucket) {
            const Code* bcol = B_small.Col(i);
            append_bytes(bcol, static_cast<std::size_t>(m_codes_bucket));
            const float* acol = a.Col(i);
            append_bytes(acol, static_cast<std::size_t>(m) * sizeof(float));
        }

        if (b.buf.size() >= bucket_flush_bytes_) {
            if (!FlushBucket(bucket_id, err)) {
                return false;
            }
        }
    }
    return true;
}

bool BaseBasicWriter::AppendBlockRawF32(std::uint64_t start_id,
                                       const std::vector<std::uint32_t>& cluster_id,
                                       const ColMajorMatrix<Code>& B_small,
                                       const ColMajorMatrix<float>& a,
                                       const float* x_f32,
                                       int ld_x_f32,
                                       std::string* err) {
    const int nblk = static_cast<int>(cluster_id.size());
    if (nblk <= 0) {
        return true;
    }
    const int m_codes = std::max(0, cfg_.m - 1);
    if (B_small.rows != m_codes || B_small.cols != nblk || a.rows != cfg_.m || a.cols != nblk) {
        if (err) {
            *err = "BaseBasicWriter::AppendBlockRawF32: B/a shape mismatch.";
        }
        return false;
    }
    if (cfg_.write_vector_bucket) {
        if (!x_f32) {
            if (err) {
                *err = "BaseBasicWriter::AppendBlockRawF32: write_vector_bucket=true but x_f32 is null.";
            }
            return false;
        }
        if (ld_x_f32 < cfg_.d) {
            if (err) {
                *err = "BaseBasicWriter::AppendBlockRawF32: invalid ld_x_f32.";
            }
            return false;
        }
        const int want = cfg_.d * static_cast<int>(sizeof(float));
        if (cfg_.raw_bytes_per_vec != want) {
            if (err) {
                *err = "BaseBasicWriter::AppendBlockRawF32: raw_bytes_per_vec != d*sizeof(float).";
            }
            return false;
        }
    }

    // cluster_id.u32 (global order).
    cluster_id_out_.write(reinterpret_cast<const char*>(cluster_id.data()),
                          static_cast<std::streamsize>(cluster_id.size() * sizeof(std::uint32_t)));
    if (!cluster_id_out_) {
        if (err) {
            *err = "BaseBasicWriter::AppendBlockRawF32: failed to write cluster_id file.";
        }
        return false;
    }

    // codes/coeffs shards (global order).
    if (!EnsureShard(start_id, err)) {
        return false;
    }
    const std::size_t n_codes =
        static_cast<std::size_t>(B_small.rows) * static_cast<std::size_t>(nblk);
    const std::size_t n_coeffs =
        static_cast<std::size_t>(a.rows) * static_cast<std::size_t>(nblk);
    codes_out_.write(reinterpret_cast<const char*>(B_small.data.data()),
                     static_cast<std::streamsize>(n_codes * sizeof(Code)));
    coeffs_out_.write(reinterpret_cast<const char*>(a.data.data()),
                      static_cast<std::streamsize>(n_coeffs * sizeof(float)));
    if (!codes_out_ || !coeffs_out_) {
        if (err) {
            *err = "BaseBasicWriter::AppendBlockRawF32: failed to write shard outputs.";
        }
        return false;
    }

    // Optional bucket record (for fast cluster-local downstream processing).
    if (!cfg_.write_vector_bucket && !cfg_.write_basic_to_bucket) {
        return true;
    }

    const int raw_bytes_per_vec = cfg_.write_vector_bucket ? cfg_.raw_bytes_per_vec : 0;
    const int m = cfg_.m;
    const int m_codes_bucket = m_codes;
    const int bucket_size = std::max(1, cfg_.bucket_size);
    const std::size_t rec_bytes =
        sizeof(std::uint32_t) + sizeof(std::uint32_t) +
        (cfg_.write_vector_bucket ? static_cast<std::size_t>(raw_bytes_per_vec) : 0ull) +
        (cfg_.write_basic_to_bucket ? (static_cast<std::size_t>(m_codes_bucket) +
                                       static_cast<std::size_t>(m) * sizeof(float))
                                    : 0ull);

    for (int i = 0; i < nblk; ++i) {
        const std::uint64_t gid = start_id + static_cast<std::uint64_t>(i);
        const std::uint32_t cid = cluster_id[static_cast<std::size_t>(i)];
        const int bucket_id = static_cast<int>(cid / static_cast<std::uint32_t>(bucket_size));
        if (!EnsureBucket(bucket_id, err)) {
            return false;
        }
        auto& b = buckets_[static_cast<std::size_t>(bucket_id)];

        if (b.buf.capacity() - b.buf.size() < rec_bytes) {
            const std::size_t need = b.buf.size() + rec_bytes;
            const std::size_t grow =
                (b.buf.capacity() > 0) ? (b.buf.capacity() + (b.buf.capacity() >> 1)) : (8ull * 1024ull * 1024ull);
            b.buf.reserve(std::max(need, grow));
        }
        auto append_bytes = [&](const void* ptr, std::size_t bytes) {
            const auto* p = static_cast<const std::uint8_t*>(ptr);
            b.buf.insert(b.buf.end(), p, p + bytes);
        };

        const auto gid32 = static_cast<std::uint32_t>(gid);
        append_bytes(&gid32, sizeof(std::uint32_t));
        append_bytes(&cid, sizeof(std::uint32_t));

        if (cfg_.write_vector_bucket) {
            const float* x = x_f32 + static_cast<std::size_t>(i) * static_cast<std::size_t>(ld_x_f32);
            append_bytes(x, static_cast<std::size_t>(raw_bytes_per_vec));
        }
        if (cfg_.write_basic_to_bucket) {
            const Code* bcol = B_small.Col(i);
            append_bytes(bcol, static_cast<std::size_t>(m_codes_bucket));
            const float* acol = a.Col(i);
            append_bytes(acol, static_cast<std::size_t>(m) * sizeof(float));
        }

        if (b.buf.size() >= bucket_flush_bytes_) {
            if (!FlushBucket(bucket_id, err)) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace stlq::io

namespace stlq::io {

namespace {

bool ReadAll(std::ifstream& in, void* dst, std::size_t bytes) {
    in.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes));
    return static_cast<std::size_t>(in.gcount()) == bytes;
}

std::string JoinPath2(const std::string& a, const std::string& b) { return JoinPath(a, b); }

}  // namespace

bool BaseBasicReader::Open(const std::string& dir, std::string* err) {
    dir_ = dir;
    meta_ = {};
    if (dir_.empty()) {
        if (err) *err = "BaseBasicReader::Open: empty dir.";
        return false;
    }

    const std::string meta_path = JoinPath2(dir_, "meta.bin");
    std::ifstream in(meta_path, std::ios::binary);
    if (!in.is_open()) {
        if (err) *err = "BaseBasicReader::Open: failed to open " + meta_path;
        return false;
    }
    struct HeaderV1 {
        std::uint32_t magic = 0;
        std::uint32_t version = 0;
        std::int32_t d = 0;
        std::int32_t m = 0;
        std::int32_t m_codes = 0;
        std::int32_t shard_size = 0;
        std::int32_t bucket_size = 0;
        std::uint32_t flags = 0;
        std::int32_t h_len = 0;
    } h;
    if (!ReadAll(in, &h, sizeof(HeaderV1))) {
        if (err) *err = "BaseBasicReader::Open: failed to read meta header.";
        return false;
    }
    if (h.magic != 0x53414243u || (h.version != 1 && h.version != 2)) {
        if (err) *err = "BaseBasicReader::Open: unsupported meta.bin version.";
        return false;
    }
    if (h.d <= 0 || h.m <= 0 || h.m_codes < 0 || h.shard_size <= 0) {
        if (err) *err = "BaseBasicReader::Open: invalid meta fields.";
        return false;
    }
    meta_.d = h.d;
    meta_.m = h.m;
    meta_.m_codes = h.m_codes;
    meta_.shard_size = h.shard_size;
    meta_.bucket_size = h.bucket_size;
    meta_.write_vector_bucket = (h.flags & 1u) != 0;
    meta_.write_basic_to_bucket = (h.flags & 2u) != 0;

    std::int32_t raw_bytes_per_vec = 0;
    if (h.version == 2) {
        if (!ReadAll(in, &raw_bytes_per_vec, sizeof(std::int32_t))) {
            if (err) *err = "BaseBasicReader::Open: failed to read raw_bytes_per_vec.";
            return false;
        }
    } else {
        raw_bytes_per_vec = meta_.write_vector_bucket ? meta_.d : 0;
    }
    if (meta_.write_vector_bucket && raw_bytes_per_vec == 0) {
        raw_bytes_per_vec = meta_.d;
    }
    meta_.raw_bytes_per_vec = static_cast<int>(std::max<std::int32_t>(0, raw_bytes_per_vec));

    if (h.h_len > 0) {
        meta_.h_vec.resize(static_cast<std::size_t>(h.h_len));
        if (!ReadAll(in, meta_.h_vec.data(), static_cast<std::size_t>(h.h_len) * sizeof(int))) {
            if (err) *err = "BaseBasicReader::Open: failed to read h_vec.";
            return false;
        }
    }
    return true;
}

bool BaseBasicReader::ReadClusterIdBlock(std::uint64_t start_id,
                                        std::uint32_t count,
                                        std::vector<std::uint32_t>* out,
                                        std::string* err) const {
    if (!out) {
        if (err) *err = "BaseBasicReader::ReadClusterIdBlock: null output.";
        return false;
    }
    out->clear();
    if (count == 0) {
        return true;
    }
    const std::filesystem::path p_u32 = std::filesystem::path(dir_) / "cluster_id.u32";
    std::ifstream in(p_u32, std::ios::binary);
    if (!in.is_open()) {
        if (err) *err = "BaseBasicReader::ReadClusterIdBlock: failed to open " + p_u32.string();
        return false;
    }
    const std::uint64_t off = start_id * sizeof(std::uint32_t);
    in.seekg(static_cast<std::streamoff>(off), std::ios::beg);
    if (!in) {
        if (err) *err = "BaseBasicReader::ReadClusterIdBlock: seek failed.";
        return false;
    }
    out->resize(count);
    in.read(reinterpret_cast<char*>(out->data()),
            static_cast<std::streamsize>(count * sizeof(std::uint32_t)));
    const std::streamsize got = in.gcount();
    const std::size_t nread = static_cast<std::size_t>(got) / sizeof(std::uint32_t);
    out->resize(nread);
    return true;
}

bool BaseBasicReader::ReadCodesBlock(std::uint64_t start_id,
                                    std::uint32_t count,
                                    ColMajorMatrix<Code>* out,
                                    std::string* err) const {
    if (!out) {
        if (err) *err = "BaseBasicReader::ReadCodesBlock: null output.";
        return false;
    }
    out->rows = meta_.m_codes;
    out->cols = 0;
    out->data.clear();
    if (count == 0) {
        return true;
    }
    const auto shard_size = static_cast<std::uint64_t>(meta_.shard_size);
    const std::uint64_t shard = start_id / shard_size;
    const std::uint64_t in_shard = start_id - shard * shard_size;
    if (in_shard + count > shard_size) {
        if (err) *err = "BaseBasicReader::ReadCodesBlock: cross-shard reads not supported yet.";
        return false;
    }
    const std::string shard_tag = ZeroPad(static_cast<int>(shard), 4);
    const std::string path = JoinPath2(dir_, "codes_shard_" + shard_tag + ".bin");
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        if (err) *err = "BaseBasicReader::ReadCodesBlock: failed to open " + path;
        return false;
    }
    const auto rec = static_cast<std::uint64_t>(meta_.m_codes);
    const std::uint64_t off = in_shard * rec;
    in.seekg(static_cast<std::streamoff>(off), std::ios::beg);
    if (!in) {
        if (err) *err = "BaseBasicReader::ReadCodesBlock: seek failed.";
        return false;
    }
    out->cols = static_cast<int>(count);
    out->data.resize(static_cast<std::size_t>(meta_.m_codes) * count);
    in.read(reinterpret_cast<char*>(out->data.data()), static_cast<std::streamsize>(out->data.size()));
    if (!in) {
        if (err) *err = "BaseBasicReader::ReadCodesBlock: read failed.";
        return false;
    }
    return true;
}

bool BaseBasicReader::ReadCoeffsBlock(std::uint64_t start_id,
                                     std::uint32_t count,
                                     ColMajorMatrix<float>* out,
                                     std::string* err) const {
    if (!out) {
        if (err) *err = "BaseBasicReader::ReadCoeffsBlock: null output.";
        return false;
    }
    out->rows = meta_.m;
    out->cols = 0;
    out->data.clear();
    if (count == 0) {
        return true;
    }
    const auto shard_size = static_cast<std::uint64_t>(meta_.shard_size);
    const std::uint64_t shard = start_id / shard_size;
    const std::uint64_t in_shard = start_id - shard * shard_size;
    if (in_shard + count > shard_size) {
        if (err) *err = "BaseBasicReader::ReadCoeffsBlock: cross-shard reads not supported yet.";
        return false;
    }
    const std::string shard_tag = ZeroPad(static_cast<int>(shard), 4);
    const std::string path = JoinPath2(dir_, "coeffs_shard_" + shard_tag + ".bin");
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        if (err) *err = "BaseBasicReader::ReadCoeffsBlock: failed to open " + path;
        return false;
    }
    const std::uint64_t rec = static_cast<std::uint64_t>(meta_.m) * sizeof(float);
    const std::uint64_t off = in_shard * rec;
    in.seekg(static_cast<std::streamoff>(off), std::ios::beg);
    if (!in) {
        if (err) *err = "BaseBasicReader::ReadCoeffsBlock: seek failed.";
        return false;
    }
    out->cols = static_cast<int>(count);
    out->data.resize(static_cast<std::size_t>(meta_.m) * count);
    in.read(reinterpret_cast<char*>(out->data.data()),
            static_cast<std::streamsize>(out->data.size() * sizeof(float)));
    if (!in) {
        if (err) *err = "BaseBasicReader::ReadCoeffsBlock: read failed.";
        return false;
    }
    return true;
}

bool BaseBasicReader::ReadCodesByIds(const std::vector<std::uint32_t>& ids,
                                    ColMajorMatrix<Code>* out,
                                    std::string* err) const {
    if (!out) {
        if (err) *err = "BaseBasicReader::ReadCodesByIds: null output.";
        return false;
    }
    out->rows = meta_.m_codes;
    out->cols = static_cast<int>(ids.size());
    out->data.resize(static_cast<std::size_t>(meta_.m_codes) * ids.size());
    if (ids.empty() || meta_.m_codes <= 0) {
        return true;
    }

    const auto shard_size = static_cast<std::uint64_t>(meta_.shard_size);
    std::uint64_t cur_shard = std::numeric_limits<std::uint64_t>::max();
    std::ifstream in;

    for (std::size_t j = 0; j < ids.size(); ++j) {
        const std::uint64_t gid = ids[j];
        const std::uint64_t shard = gid / shard_size;
        const std::uint64_t in_shard = gid - shard * shard_size;
        if (shard != cur_shard) {
            cur_shard = shard;
            if (in.is_open()) {
                in.close();
            }
            const std::string shard_tag = ZeroPad(static_cast<int>(shard), 4);
            const std::string path = JoinPath2(dir_, "codes_shard_" + shard_tag + ".bin");
            in.open(path, std::ios::binary);
            if (!in.is_open()) {
                if (err) *err = "BaseBasicReader::ReadCodesByIds: failed to open " + path;
                return false;
            }
        }
        const std::uint64_t off = in_shard * static_cast<std::uint64_t>(meta_.m_codes);
        in.seekg(static_cast<std::streamoff>(off), std::ios::beg);
        if (!in) {
            if (err) *err = "BaseBasicReader::ReadCodesByIds: seek failed.";
            return false;
        }
        Code* dst = out->Col(static_cast<int>(j));
        in.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(meta_.m_codes));
        if (!in) {
            if (err) *err = "BaseBasicReader::ReadCodesByIds: read failed.";
            return false;
        }
    }
    return true;
}

bool BaseBasicReader::ReadCoeffsByIds(const std::vector<std::uint32_t>& ids,
                                     ColMajorMatrix<float>* out,
                                     std::string* err) const {
    if (!out) {
        if (err) *err = "BaseBasicReader::ReadCoeffsByIds: null output.";
        return false;
    }
    out->rows = meta_.m;
    out->cols = static_cast<int>(ids.size());
    out->data.resize(static_cast<std::size_t>(meta_.m) * ids.size());
    if (ids.empty() || meta_.m <= 0) {
        return true;
    }

    const auto shard_size = static_cast<std::uint64_t>(meta_.shard_size);
    std::uint64_t cur_shard = std::numeric_limits<std::uint64_t>::max();
    std::ifstream in;

    const std::uint64_t rec = static_cast<std::uint64_t>(meta_.m) * sizeof(float);

    for (std::size_t j = 0; j < ids.size(); ++j) {
        const std::uint64_t gid = ids[j];
        const std::uint64_t shard = gid / shard_size;
        const std::uint64_t in_shard = gid - shard * shard_size;
        if (shard != cur_shard) {
            cur_shard = shard;
            if (in.is_open()) {
                in.close();
            }
            const std::string shard_tag = ZeroPad(static_cast<int>(shard), 4);
            const std::string path = JoinPath2(dir_, "coeffs_shard_" + shard_tag + ".bin");
            in.open(path, std::ios::binary);
            if (!in.is_open()) {
                if (err) *err = "BaseBasicReader::ReadCoeffsByIds: failed to open " + path;
                return false;
            }
        }
        const std::uint64_t off = in_shard * rec;
        in.seekg(static_cast<std::streamoff>(off), std::ios::beg);
        if (!in) {
            if (err) *err = "BaseBasicReader::ReadCoeffsByIds: seek failed.";
            return false;
        }
        float* dst = out->Col(static_cast<int>(j));
        in.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(rec));
        if (!in) {
            if (err) *err = "BaseBasicReader::ReadCoeffsByIds: read failed.";
            return false;
        }
    }
    return true;
}

}  // namespace stlq::io
