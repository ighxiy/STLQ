#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "stlq/common/types.h"

namespace stlq::io {

// List-order contiguous base encodings aligned with IVF CSR ids.
// Layout:
// - codes.bin:  u8 codes for layers 1..m-1, column-major (m_codes x ntotal)
// - coeffs.bin: f32 coeffs for layers 0..m-1, column-major (m x ntotal)
// The root layer (layer 0) code is the IVF list id (cluster id), not stored here.
struct BaseListMeta {
    int d = 0;
    int m = 0;
    int m_codes = 0;
    std::uint64_t ntotal = 0;
    bool has_raw_u8 = false;
    int raw_u8_bytes_per_vec = 0;
    bool has_raw_f32 = false;
    int raw_f32_bytes_per_vec = 0;
};

class BaseListReader {
public:
    bool Open(const std::string& dir, std::string* err);

    const BaseListMeta& meta() const { return meta_; }
    const std::string& dir() const { return dir_; }
    bool HasRawU8() const { return meta_.has_raw_u8; }
    bool HasRawF32() const { return meta_.has_raw_f32; }

    bool ReadCodesSpan(std::uint64_t begin,
                       std::uint32_t count,
                       std::vector<Code>* out,
                       std::string* err) const;
    bool ReadCoeffsSpan(std::uint64_t begin,
                        std::uint32_t count,
                        std::vector<float>* out,
                        std::string* err) const;
    bool ReadRawU8Span(std::uint64_t begin,
                       std::uint32_t count,
                       ColMajorMatrix<std::uint8_t>* out,
                       std::string* err) const;
    bool ReadRawF32Span(std::uint64_t begin,
                        std::uint32_t count,
                        ColMajorMatrix<float>* out,
                        std::string* err) const;

private:
    std::string dir_;
    BaseListMeta meta_;
};

// Thread-local (or single-thread) reader that keeps file handles open to avoid open/close per span.
// This reader is not thread-safe for concurrent calls; create one instance per thread.
class BaseListThreadReader {
public:
    BaseListThreadReader() = default;
    ~BaseListThreadReader();

    bool OpenFrom(const BaseListReader& src, std::string* err);
    bool Open(const std::string& dir, std::string* err);

    const BaseListMeta& meta() const { return meta_; }
    const std::string& dir() const { return dir_; }
    bool HasRawU8() const { return meta_.has_raw_u8; }
    bool HasRawF32() const { return meta_.has_raw_f32; }

    bool ReadCodesSpan(std::uint64_t begin,
                       std::uint32_t count,
                       std::vector<Code>* out,
                       std::string* err);
    bool ReadCoeffsSpan(std::uint64_t begin,
                        std::uint32_t count,
                        std::vector<float>* out,
                        std::string* err);
    bool ReadRawU8Span(std::uint64_t begin,
                       std::uint32_t count,
                       ColMajorMatrix<std::uint8_t>* out,
                       std::string* err);
    bool ReadRawF32Span(std::uint64_t begin,
                        std::uint32_t count,
                        ColMajorMatrix<float>* out,
                        std::string* err);

private:
    bool OpenFiles(std::string* err);
    void Close();

    std::string dir_;
    BaseListMeta meta_;
    std::ifstream codes_in_;
    std::ifstream coeffs_in_;
    std::ifstream raw_u8_in_;
    std::ifstream raw_f32_in_;
};

// Build list-order contiguous `codes.bin` and `coeffs.bin` from the per-cluster bucket files
// produced by `BaseBasicWriter` (requires `write_basic_to_bucket=true`).
//
// This avoids random seeks to global shards and allows sequential list scanning.
bool BuildBaseListStoreFromBasicBuckets(const std::string& base_basic_dir,
                                        const class IvfListsReader& lists,
                                        const std::string& out_dir,
                                        const std::string& tmp_dir,
                                        bool keep_tmp,
                                        bool profile_timing,
                                        std::string* err);

}  // namespace stlq::io
