#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

#include "stlq/common/types.h"

namespace stlq::io {

// `.fvecs` reader (float32 vectors):
// Each record: int32 d + float[d]
class FvecsReader {
public:
    bool Open(const std::string& path, std::string* err);

    int d() const { return d_; }
    std::uint64_t n() const { return n_; }
    std::uint64_t record_bytes() const { return rec_; }
    const std::string& path() const { return path_; }

    bool ReadBlock(std::uint64_t start,
                   std::uint32_t count,
                   ColMajorMatrix<float>* out,
                   std::string* err) const;

    // Best-effort: read `count` vectors starting at `start` directly into a caller-provided
    // column-major buffer (ld == d). This avoids allocating an intermediate matrix.
    bool ReadBlockInto(std::uint64_t start,
                       std::uint32_t count,
                       float* dst_colmajor,
                       std::size_t dst_bytes,
                       std::string* err) const;

    // Slow random gather reads (random seeks). `ids` are 0-based vector indices.
    // Intended for fallback (small data / regression) when list-order raw stores are not available.
    bool ReadByIds(const std::vector<std::uint32_t>& ids,
                   ColMajorMatrix<float>* out,
                   std::string* err) const;

private:
    std::string path_;
    int d_ = 0;
    std::uint64_t n_ = 0;
    std::uint64_t rec_ = 0;
    // Reused scratch buffer for sequential block reads (avoids per-call allocation/zero-fill).
    mutable std::vector<std::uint8_t> scratch_;
};

}  // namespace stlq::io
