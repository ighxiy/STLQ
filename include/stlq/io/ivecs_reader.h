#pragma once

#include <cstdint>
#include <string>

#include "stlq/common/types.h"

namespace stlq::io {

// BigANN `.ivecs` reader (int32 vectors):
// Each record: int32 k + int32[k]
class IvecsReader {
public:
    bool Open(const std::string& path, std::string* err);

    int k() const { return k_; }
    std::uint64_t n() const { return n_; }
    std::uint64_t record_bytes() const { return rec_; }
    const std::string& path() const { return path_; }

    bool ReadBlock(std::uint64_t start,
                   std::uint32_t count,
                   ColMajorMatrix<std::int32_t>* out,
                   std::string* err) const;

private:
    std::string path_;
    int k_ = 0;
    std::uint64_t n_ = 0;
    std::uint64_t rec_ = 0;
    // Reused scratch buffer for sequential block reads (avoids per-call allocation/zero-fill).
    mutable std::vector<std::uint8_t> scratch_;
};

}  // namespace stlq::io
