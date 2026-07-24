#pragma once

#include <cstdint>
#include <cstddef>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "stlq/common/types.h"

namespace stlq::io {

// BigANN `.bvecs` reader (uint8 vectors):
// Each record: int32 d + uint8[d]
class BvecsReader {
public:
    bool Open(const std::string& path, std::string* err);

    int d() const { return d_; }
    std::uint64_t n() const { return n_; }
    std::uint64_t record_bytes() const { return rec_; }
    const std::string& path() const { return path_; }

    bool ReadBlock(std::uint64_t start,
                   std::uint32_t count,
                   ColMajorMatrix<std::uint8_t>* out,
                   std::string* err) const;

    // Best-effort: read `count` vectors starting at `start` directly into a caller-provided
    // column-major buffer (ld == d). This avoids allocating an intermediate matrix.
    bool ReadBlockInto(std::uint64_t start,
                       std::uint32_t count,
                       std::uint8_t* dst_colmajor,
                       std::size_t dst_bytes,
                       std::string* err) const;

    // Best-effort timing for the most recent ReadBlock*/ReadBlockInto call.
    double last_read_file_s() const { return last_read_file_s_; }
    double last_unpack_s() const { return last_unpack_s_; }

#ifndef NDEBUG
    struct DebugStats {
        std::uint64_t read_block_calls = 0;
        std::uint64_t read_block_into_calls = 0;
        std::uint64_t seek_calls = 0;
        std::uint64_t sequential_seek_skips = 0;
        std::uint64_t bytes_requested = 0;
        std::uint64_t bytes_read = 0;
        std::uint64_t last_offset_bytes = 0;
        std::uint64_t last_read_bytes = 0;
    };
    DebugStats debug_stats() const { return dbg_; }
#endif

    bool ReadByIds(const std::vector<std::uint32_t>& ids,
                   ColMajorMatrix<std::uint8_t>* out,
                   std::string* err) const;

    bool ReadOne(std::uint64_t id,
                 std::uint8_t* out_vec,
                 std::string* err) const;

private:
    std::string path_;
    int d_ = 0;
    std::uint64_t n_ = 0;
    std::uint64_t rec_ = 0;
    // Keep the file open to avoid per-block open/close overhead in streaming paths.
    // Not thread-safe; our pipeline does not call ReadBlock* concurrently on the same reader.
    mutable std::ifstream in_;
    mutable double last_read_file_s_ = 0.0;
    mutable double last_unpack_s_ = 0.0;
	// Reused scratch buffer for sequential block reads (avoids per-call allocation/zero-fill).
	// Safe because our current pipeline never calls ReadBlock concurrently on the same reader.
	//
	// NOTE: use a raw buffer (not std::vector) to avoid per-call value-initialization when resizing.
    mutable std::unique_ptr<std::uint8_t[]> scratch_;
    mutable std::size_t scratch_cap_ = 0;
    // Best-effort sequential cursor: avoids seekg() when callers read consecutive blocks.
    // Safe because our pipeline does not call ReadBlock* concurrently on the same reader.
    mutable std::uint64_t cursor_off_ = 0;
    mutable bool cursor_valid_ = false;

#ifndef NDEBUG
    mutable DebugStats dbg_{};
#endif
};

}  // namespace stlq::io
