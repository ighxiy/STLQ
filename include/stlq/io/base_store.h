#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "stlq/common/types.h"

namespace stlq::io {

struct BaseBasicStoreConfig {
    std::string dir;
    int d = 0;
    int m = 0;
    std::vector<int> h_vec;
    int shard_size = 2000000;
    int bucket_size = 256;
    int bucket_flush_mb = 256;
    bool write_vector_bucket = true;
    bool write_basic_to_bucket = true;
    // Raw vector bytes stored in bucket records when `write_vector_bucket=true`.
    // - 0 means "default" (u8, `d` bytes per vector) for backward compatibility.
    // - For float datasets, set to `d * sizeof(float)` to store raw f32 without lossy clamping.
    int raw_bytes_per_vec = 0;
};

class BaseBasicWriter {
public:
    ~BaseBasicWriter();
    bool Open(const BaseBasicStoreConfig& cfg, std::string* err);
    // Resume writing into an existing BaseBasicStore that has been truncated to a safe
    // checkpoint boundary by the caller.
    // - `start_id` is the next global id to write.
    // - `bucket_sizes` are the on-disk committed sizes (bytes) for each bucket file at `start_id`.
    bool OpenResume(const BaseBasicStoreConfig& cfg,
                    std::uint64_t start_id,
                    const std::vector<std::uint64_t>& bucket_sizes,
                    std::string* err);
    bool Close(std::string* err);
    // Flush all output streams (best-effort durability; no fsync).
    bool Flush(std::string* err);

    // Committed on-disk bytes per bucket file (excludes in-memory buffers).
    const std::vector<std::uint64_t>& bucket_file_bytes() const { return bucket_file_bytes_; }

    bool AppendBlock(std::uint64_t start_id,
                     const std::vector<std::uint32_t>& cluster_id,
                     const ColMajorMatrix<Code>& B_small,
                     const ColMajorMatrix<float>& a,
                     const ColMajorMatrix<std::uint8_t>* x_u8_optional,
                     std::string* err);

    // Variant for streaming readers that already have a column-major u8 tile in memory.
    // Avoids materializing a temporary `ColMajorMatrix<uint8_t>` per tile.
    bool AppendBlockRawU8(std::uint64_t start_id,
                          const std::vector<std::uint32_t>& cluster_id,
                          const ColMajorMatrix<Code>& B_small,
                          const ColMajorMatrix<float>& a,
                          const std::uint8_t* x_u8,
                          int ld_x_u8,
                          std::string* err);

    // Variant for float datasets: store raw f32 vectors in bucket records.
    // `x_f32` points to a column-major tile with leading dimension `ld_x_f32` (in floats).
    bool AppendBlockRawF32(std::uint64_t start_id,
                           const std::vector<std::uint32_t>& cluster_id,
                           const ColMajorMatrix<Code>& B_small,
                           const ColMajorMatrix<float>& a,
                           const float* x_f32,
                           int ld_x_f32,
                           std::string* err);

private:
    bool EnsureShard(std::uint64_t global_id, std::string* err);
    bool EnsureBucket(int bucket_id, std::string* err);
    bool FlushBucket(int bucket_id, std::string* err);

    BaseBasicStoreConfig cfg_;
    std::ofstream cluster_id_out_;

    int current_shard_ = -1;
    std::uint64_t current_shard_begin_ = 0;
    std::ofstream codes_out_;
    std::ofstream coeffs_out_;

    bool resume_mode_ = false;
    std::uint64_t resume_start_id_ = 0;
    int resume_shard_ = 0;
    int resume_in_shard_ = 0;

    struct BucketState {
        bool open = false;
        std::ofstream out;
        std::vector<std::uint8_t> buf;
    };
    std::vector<BucketState> buckets_;
    std::vector<std::uint64_t> bucket_file_bytes_;
    std::size_t bucket_flush_bytes_ = 0;
};

struct BaseBasicMeta {
    int d = 0;
    int m = 0;
    int m_codes = 0;
    int shard_size = 0;
    int bucket_size = 0;
    bool write_vector_bucket = false;
    bool write_basic_to_bucket = false;
    int raw_bytes_per_vec = 0;
    std::vector<int> h_vec;
};

class BaseBasicReader {
public:
    bool Open(const std::string& dir, std::string* err);

    const BaseBasicMeta& meta() const { return meta_; }
    const std::string& dir() const { return dir_; }

    bool ReadClusterIdBlock(std::uint64_t start_id,
                            std::uint32_t count,
                            std::vector<std::uint32_t>* out,
                            std::string* err) const;

    bool ReadCodesBlock(std::uint64_t start_id,
                        std::uint32_t count,
                        ColMajorMatrix<Code>* out,
                        std::string* err) const;

    bool ReadCoeffsBlock(std::uint64_t start_id,
                         std::uint32_t count,
                         ColMajorMatrix<float>* out,
                         std::string* err) const;

    // Slow but simple gather reads (random seeks). `ids` are global ids.
    // Intended for fallback (CSR lists + random read) until bucket-order contiguous stores are built.
    bool ReadCodesByIds(const std::vector<std::uint32_t>& ids,
                        ColMajorMatrix<Code>* out,
                        std::string* err) const;

    bool ReadCoeffsByIds(const std::vector<std::uint32_t>& ids,
                         ColMajorMatrix<float>* out,
                         std::string* err) const;

private:
    std::string dir_;
    BaseBasicMeta meta_;
};

}  // namespace stlq::io
