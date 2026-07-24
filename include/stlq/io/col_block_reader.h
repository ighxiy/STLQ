#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include "stlq/common/types.h"

namespace stlq::io {

enum class ColBlockDType {
    kF32 = 0,
    kU8 = 1,
};

struct ColBlock {
    ColBlockDType dtype = ColBlockDType::kF32;
    // Column-major float32 block: data layout matches ColMajorMatrix<float>.
    ColMajorMatrix<float> X_f32;  // (d x cols)
    // Column-major uint8 block: data layout matches ColMajorMatrix<uint8_t>.
    ColMajorMatrix<std::uint8_t> X_u8;  // (d x cols)
    std::int64_t col0 = 0;    // global starting column index

    const ColMajorMatrix<float>& X() const {
        if (dtype != ColBlockDType::kF32) {
            throw std::runtime_error("ColBlock::X(): dtype is not f32.");
        }
        return X_f32;
    }
};

class IColBlockReader {
public:
    virtual ~IColBlockReader() = default;
    virtual int d() const = 0;
    virtual std::int64_t n() const = 0;
    // Best-effort: payload dtype produced by this reader (constant for all blocks).
    // Default is float32.
    virtual ColBlockDType dtype() const { return ColBlockDType::kF32; }
    virtual bool Reset(std::string* err) = 0;
    // Best-effort: reposition the internal cursor to `col0`.
    // Default implementation returns false (unsupported).
    virtual bool Seek(std::int64_t /*col0*/, std::string* /*err*/) { return false; }
    // Reads up to `max_cols` columns. Returns false on hard error.
    // On EOF, returns true with the selected payload having cols == 0.
    virtual bool ReadNext(int max_cols, ColBlock* out, std::string* err) = 0;

    // Best-effort: read the next block directly into a caller-provided buffer.
    // This is intended for GPU streaming paths where `dst` points to pinned host memory
    // suitable for async H2D. The buffer must be column-major packed with ld == d().
    //
    // Returns true if the reader supports the direct path. On success, `*out_col0` and `*out_cols`
    // describe the block. On EOF, returns true with `*out_cols == 0`.
    //
    // Default implementation returns false (unsupported).
    virtual bool ReadNextInto(int /*max_cols*/,
                              void* /*dst*/,
                              std::size_t /*dst_bytes*/,
                              std::int64_t* /*out_col0*/,
                              int* /*out_cols*/,
                              std::string* /*err*/) {
        return false;
    }
};

// Optional timing breakdown for readers (best-effort; intended for profiling only).
struct ColBlockReaderTimingBreakdown {
    double file_s = 0.0;   // file seek+read time
    double unpack_s = 0.0; // unpack/transpose/convert time into dst/out block
};

class IColBlockReaderTiming {
public:
    virtual ~IColBlockReaderTiming() = default;
    virtual ColBlockReaderTimingBreakdown LastTimingBreakdown() const = 0;
};

#ifndef NDEBUG
// Optional debug stats for readers (best-effort; debug builds only).
struct ColBlockReaderDebugStats {
    std::uint64_t read_calls = 0;
    std::uint64_t seek_calls = 0;
    std::uint64_t sequential_seek_skips = 0;
    std::uint64_t bytes_requested = 0;
    std::uint64_t bytes_read = 0;
    std::uint64_t last_offset_bytes = 0;
    std::uint64_t last_read_bytes = 0;
};

class IColBlockReaderDebugStats {
public:
    virtual ~IColBlockReaderDebugStats() = default;
    virtual ColBlockReaderDebugStats DebugStats() const = 0;
};
#endif

// Wrap an in-memory matrix and yield blocks by copying slices.
std::unique_ptr<IColBlockReader> MakeMatrixColBlockReader(const ColMajorMatrix<float>& X);

// Wrap a .fvecs file reader and yield float32 blocks (no normalization).
std::unique_ptr<IColBlockReader> MakeFvecsColBlockReader(const std::string& path,
                                                         std::string* err);

// Wrap a .fbin file reader (Yandex flat binary) and yield float32 blocks (no normalization).
std::unique_ptr<IColBlockReader> MakeFbinColBlockReader(const std::string& path,
                                                        std::string* err);

// Wrap a .bvecs/.siftbin-like reader and yield float32 blocks (uint8->float conversion only).
std::unique_ptr<IColBlockReader> MakeBvecsColBlockReader(const std::string& path,
                                                         std::string* err);

// Wrap a .bvecs/.siftbin-like reader and yield uint8 blocks (no CPU conversion).
std::unique_ptr<IColBlockReader> MakeBvecsColBlockReaderU8(const std::string& path,
                                                           std::string* err);

// Wrap an existing reader with a Host RAM cache.
// If caching is successful (alloc ok), subsequent passes serve data from RAM.
// If allocation fails, it falls back to pass-through (or returns nullptr if fatal only if strict).
// Ownership of `base` is transferred to the wrapper.
std::unique_ptr<IColBlockReader> MakeCachedColBlockReader(std::unique_ptr<IColBlockReader> base,
                                                          std::string* err);

}  // namespace stlq::io
