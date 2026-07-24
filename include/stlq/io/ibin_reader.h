#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "stlq/common/types.h"

namespace stlq::io {

// Yandex `.ibin` reader (int32 flat binary):
// File layout: int32 nrows + int32 ncols + nrows*ncols int32 (row-major, no per-row dim prefix).
// Used for groundtruth files in Deep1B/Deep10M benchmarks.
class IbinReader {
public:
    bool Open(const std::string& path, std::string* err);

    int k() const { return k_; }
    std::uint64_t n() const { return n_; }
    const std::string& path() const { return path_; }

    bool ReadBlock(std::uint64_t start,
                   std::uint32_t count,
                   ColMajorMatrix<std::int32_t>* out,
                   std::string* err) const;

private:
    static constexpr std::uint64_t kHeaderBytes = 8;  // 2 x int32

    std::string path_;
    int k_ = 0;
    std::uint64_t n_ = 0;
    mutable std::vector<std::uint8_t> scratch_;
};

}  // namespace stlq::io
