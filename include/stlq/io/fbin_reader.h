#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

#include "stlq/common/types.h"

namespace stlq::io {

// Yandex `.fbin` reader (float32 flat binary):
// File layout: int32 nrows + int32 ncols + nrows*ncols float32 (row-major, no per-row dim prefix).
// Used by Deep1B/Deep10M and similar benchmarks.
class FbinReader {
public:
    bool Open(const std::string& path, std::string* err);

    int d() const { return d_; }
    std::uint64_t n() const { return n_; }
    const std::string& path() const { return path_; }

    bool ReadBlock(std::uint64_t start,
                   std::uint32_t count,
                   ColMajorMatrix<float>* out,
                   std::string* err) const;

    bool ReadBlockInto(std::uint64_t start,
                       std::uint32_t count,
                       float* dst_colmajor,
                       std::size_t dst_bytes,
                       std::string* err) const;

    bool ReadByIds(const std::vector<std::uint32_t>& ids,
                   ColMajorMatrix<float>* out,
                   std::string* err) const;

private:
    static constexpr std::uint64_t kHeaderBytes = 8;  // 2 x int32

    std::string path_;
    int d_ = 0;
    std::uint64_t n_ = 0;
    // Reused scratch buffer for sequential block reads.
    mutable std::vector<std::uint8_t> scratch_;
};

}  // namespace stlq::io
