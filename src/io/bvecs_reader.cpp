#include "stlq/io/bvecs_reader.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace stlq::io {

namespace {

bool ReadInt32(std::ifstream& in, std::int32_t* out) {
    std::int32_t v = 0;
    in.read(reinterpret_cast<char*>(&v), sizeof(std::int32_t));
    if (!in) {
        return false;
    }
    *out = v;
    return true;
}

bool ReadRecordAt(std::ifstream& in,
                  std::uint64_t offset_bytes,
                  int d,
                  std::uint8_t* out_vec,
                  std::string* err) {
    in.clear();
    in.seekg(static_cast<std::streamoff>(offset_bytes), std::ios::beg);
    if (!in) {
        if (err) {
            *err = "BvecsReader: seek failed.";
        }
        return false;
    }
    std::int32_t dim = 0;
    if (!ReadInt32(in, &dim)) {
        if (err) {
            *err = "BvecsReader: failed to read dim.";
        }
        return false;
    }
    if (dim != d) {
        if (err) {
            *err = "BvecsReader: dim mismatch (expected " + std::to_string(d) + ", got " +
                   std::to_string(dim) + ")";
        }
        return false;
    }
    in.read(reinterpret_cast<char*>(out_vec), d);
    if (!in) {
        if (err) {
            *err = "BvecsReader: failed to read payload.";
        }
        return false;
    }
    return true;
}

}  // namespace

bool BvecsReader::Open(const std::string& path, std::string* err) {
    path_ = path;
    d_ = 0;
    n_ = 0;
    rec_ = 0;
    scratch_.reset();
    scratch_cap_ = 0;
    cursor_off_ = 0;
    cursor_valid_ = false;
#ifndef NDEBUG
    dbg_ = {};
#endif
    if (in_.is_open()) {
        in_.close();
    }

    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        if (err) {
            *err = "BvecsReader: failed to open file: " + path;
        }
        return false;
    }

    std::int32_t dim = 0;
    if (!ReadInt32(in, &dim)) {
        if (err) {
            *err = "BvecsReader: failed to read dim from: " + path;
        }
        return false;
    }
    if (dim <= 0) {
        if (err) {
            *err = "BvecsReader: invalid dim in: " + path;
        }
        return false;
    }

    std::uint64_t size = 0;
    try {
        size = static_cast<std::uint64_t>(std::filesystem::file_size(path));
    } catch (...) {
        if (err) {
            *err = "BvecsReader: failed to stat file: " + path;
        }
        return false;
    }

    const std::uint64_t rec = static_cast<std::uint64_t>(sizeof(std::int32_t)) + static_cast<std::uint64_t>(dim);
    if (rec == 0) {
        if (err) {
            *err = "BvecsReader: invalid record size for: " + path;
        }
        return false;
    }

    const std::uint64_t n = size / rec;
    if (n == 0) {
        if (err) {
            *err = "BvecsReader: empty file: " + path;
        }
        return false;
    }

    d_ = dim;
    rec_ = rec;
    n_ = n;

    in_.open(path_, std::ios::binary);
    if (!in_.is_open()) {
        if (err) *err = "BvecsReader: failed to open file for streaming: " + path_;
        return false;
    }
    return true;
}

bool BvecsReader::ReadBlock(std::uint64_t start,
                            std::uint32_t count,
                            ColMajorMatrix<std::uint8_t>* out,
                            std::string* err) const {
    if (!out) {
        if (err) {
            *err = "BvecsReader::ReadBlock: null output.";
        }
        return false;
    }
    out->rows = 0;
    out->cols = 0;
    out->data.clear();
    if (d_ <= 0 || rec_ == 0 || n_ == 0) {
        if (err) {
            *err = "BvecsReader::ReadBlock: reader not open.";
        }
        return false;
    }
    if (start >= n_ || count == 0) {
        return true;
    }
    const std::uint64_t avail = n_ - start;
    const std::uint32_t nread = static_cast<std::uint32_t>(std::min<std::uint64_t>(avail, count));
    // NOTE: the streaming pipeline keeps `in_` open after Open(). Avoid redundant open/close checks here.
#ifndef NDEBUG
    if (!in_.is_open()) {
        if (err) *err = "BvecsReader::ReadBlock: file not open.";
        return false;
    }
#endif

    const std::uint64_t offset = start * rec_;
    const std::uint64_t bytes = static_cast<std::uint64_t>(nread) * rec_;
#ifndef NDEBUG
    dbg_.read_block_calls += 1;
    dbg_.bytes_requested += bytes;
    dbg_.last_offset_bytes = offset;
    dbg_.last_read_bytes = bytes;
#endif
    if (scratch_cap_ < static_cast<std::size_t>(bytes)) {
        scratch_.reset(new std::uint8_t[static_cast<std::size_t>(bytes)]);
        scratch_cap_ = static_cast<std::size_t>(bytes);
    }
    const auto t_file0 = std::chrono::high_resolution_clock::now();
    in_.clear();
    // 2A: sequential fast-path (skip seekg when already at the right offset).
    if (!cursor_valid_ || cursor_off_ != offset) {
#ifndef NDEBUG
        dbg_.seek_calls += 1;
#endif
        in_.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        if (!in_) {
            if (err) {
                *err = "BvecsReader::ReadBlock: seek failed.";
            }
            return false;
        }
        cursor_off_ = offset;
        cursor_valid_ = true;
    } else {
#ifndef NDEBUG
        dbg_.sequential_seek_skips += 1;
#endif
    }
    in_.read(reinterpret_cast<char*>(scratch_.get()), static_cast<std::streamsize>(bytes));
    if (!in_) {
        if (err) {
            *err = "BvecsReader::ReadBlock: read failed.";
        }
        return false;
    }
#ifndef NDEBUG
    dbg_.bytes_read += bytes;
#endif
    cursor_off_ += bytes;
    const auto t_file1 = std::chrono::high_resolution_clock::now();
    last_read_file_s_ = std::chrono::duration<double>(t_file1 - t_file0).count();

    out->rows = d_;
    out->cols = static_cast<int>(nread);
    out->data.resize(static_cast<std::size_t>(d_) * nread);

    // Validate dimension once (BigANN format has constant d per record).
#ifndef NDEBUG
    {
        std::int32_t dim0 = 0;
        std::memcpy(&dim0, scratch_.get(), sizeof(std::int32_t));
        if (dim0 != d_) {
            if (err) {
                *err = "BvecsReader::ReadBlock: dim mismatch (expected " + std::to_string(d_) +
                       ", got " + std::to_string(dim0) + ")";
            }
            return false;
        }
    }
#endif
    const std::uint8_t* src = scratch_.get() + sizeof(std::int32_t);
    const auto payload_stride = static_cast<std::size_t>(rec_);
    const auto payload_bytes = static_cast<std::size_t>(d_);
    const auto t_unpack0 = std::chrono::high_resolution_clock::now();
    if (nread >= 4096) {
        #pragma omp parallel for default(none) schedule(static) shared(out, src, payload_stride, payload_bytes, nread)
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(nread); ++i) {
            std::uint8_t* dst = out->Col(static_cast<int>(i));
            const std::uint8_t* rec_ptr = src + static_cast<std::size_t>(i) * payload_stride;
            std::memcpy(dst, rec_ptr, payload_bytes);
        }
    } else {
        for (std::uint32_t i = 0; i < nread; ++i) {
            std::uint8_t* dst = out->Col(static_cast<int>(i));
            const std::uint8_t* rec_ptr = src + static_cast<std::size_t>(i) * payload_stride;
            std::memcpy(dst, rec_ptr, payload_bytes);
        }
    }
    const auto t_unpack1 = std::chrono::high_resolution_clock::now();
    last_unpack_s_ = std::chrono::duration<double>(t_unpack1 - t_unpack0).count();
    return true;
}

bool BvecsReader::ReadBlockInto(std::uint64_t start,
                                std::uint32_t count,
                                std::uint8_t* dst_colmajor,
                                std::size_t dst_bytes,
                                std::string* err) const {
    // NOTE: keep structure but drop hot null/bounds checks in release builds.
#ifndef NDEBUG
    if (!dst_colmajor) {
        if (err) *err = "BvecsReader::ReadBlockInto: null dst.";
        return false;
    }
#endif
    if (d_ <= 0 || rec_ == 0 || n_ == 0) {
        if (err) *err = "BvecsReader::ReadBlockInto: reader not open.";
        return false;
    }
    if (count == 0 || start >= n_) {
        return true;
    }
    const std::uint64_t avail = n_ - start;
    const std::uint32_t nread = static_cast<std::uint32_t>(std::min<std::uint64_t>(avail, count));
#ifndef NDEBUG
    const std::size_t need = static_cast<std::size_t>(d_) * static_cast<std::size_t>(nread);
    if (dst_bytes < need) {
        if (err) *err = "BvecsReader::ReadBlockInto: dst_bytes too small.";
        return false;
    }
    if (!in_.is_open()) {
        if (err) *err = "BvecsReader::ReadBlockInto: file not open.";
        return false;
    }
#else
    (void)dst_bytes;
#endif

    const std::uint64_t offset = start * rec_;
    const std::uint64_t bytes = static_cast<std::uint64_t>(nread) * rec_;
#ifndef NDEBUG
    dbg_.read_block_into_calls += 1;
    dbg_.bytes_requested += bytes;
    dbg_.last_offset_bytes = offset;
    dbg_.last_read_bytes = bytes;
#endif
    if (scratch_cap_ < static_cast<std::size_t>(bytes)) {
        scratch_.reset(new std::uint8_t[static_cast<std::size_t>(bytes)]);
        scratch_cap_ = static_cast<std::size_t>(bytes);
    }
    const auto t_file0 = std::chrono::high_resolution_clock::now();
    in_.clear();
    // 2A: sequential fast-path (skip seekg when already at the right offset).
    if (!cursor_valid_ || cursor_off_ != offset) {
#ifndef NDEBUG
        dbg_.seek_calls += 1;
#endif
        in_.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        if (!in_) {
            if (err) *err = "BvecsReader::ReadBlockInto: seek failed.";
            return false;
        }
        cursor_off_ = offset;
        cursor_valid_ = true;
    } else {
#ifndef NDEBUG
        dbg_.sequential_seek_skips += 1;
#endif
    }
    in_.read(reinterpret_cast<char*>(scratch_.get()), static_cast<std::streamsize>(bytes));
    if (!in_) {
        if (err) *err = "BvecsReader::ReadBlockInto: read failed.";
        return false;
    }
#ifndef NDEBUG
    dbg_.bytes_read += bytes;
#endif
    cursor_off_ += bytes;
    const auto t_file1 = std::chrono::high_resolution_clock::now();
    last_read_file_s_ = std::chrono::duration<double>(t_file1 - t_file0).count();

    // Validate dimension once (BigANN format has constant d per record).
#ifndef NDEBUG
    {
        std::int32_t dim0 = 0;
        std::memcpy(&dim0, scratch_.get(), sizeof(std::int32_t));
        if (dim0 != d_) {
            if (err) {
                *err = "BvecsReader::ReadBlockInto: dim mismatch (expected " + std::to_string(d_) +
                       ", got " + std::to_string(dim0) + ")";
            }
            return false;
        }
    }
#endif
    const std::uint8_t* src = scratch_.get() + sizeof(std::int32_t);
    const auto payload_stride = static_cast<std::size_t>(rec_);
    const auto payload_bytes = static_cast<std::size_t>(d_);
    const auto t_unpack0 = std::chrono::high_resolution_clock::now();
    if (nread >= 4096) {
        #pragma omp parallel for default(none) schedule(static) shared(dst_colmajor, src, payload_stride, payload_bytes, nread)
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(nread); ++i) {
            std::uint8_t* dst = dst_colmajor + static_cast<std::size_t>(i) * payload_bytes;
            const std::uint8_t* rec_ptr = src + static_cast<std::size_t>(i) * payload_stride;
            std::memcpy(dst, rec_ptr, payload_bytes);
        }
    } else {
        for (std::uint32_t i = 0; i < nread; ++i) {
            std::uint8_t* dst = dst_colmajor + static_cast<std::size_t>(i) * payload_bytes;
            const std::uint8_t* rec_ptr = src + static_cast<std::size_t>(i) * payload_stride;
            std::memcpy(dst, rec_ptr, payload_bytes);
        }
    }
    const auto t_unpack1 = std::chrono::high_resolution_clock::now();
    last_unpack_s_ = std::chrono::duration<double>(t_unpack1 - t_unpack0).count();
    return true;
}

bool BvecsReader::ReadByIds(const std::vector<std::uint32_t>& ids,
                            ColMajorMatrix<std::uint8_t>* out,
                            std::string* err) const {
    if (!out) {
        if (err) {
            *err = "BvecsReader::ReadByIds: null output.";
        }
        return false;
    }
    out->rows = 0;
    out->cols = 0;
    out->data.clear();
    if (d_ <= 0 || rec_ == 0 || n_ == 0) {
        if (err) {
            *err = "BvecsReader::ReadByIds: reader not open.";
        }
        return false;
    }
    if (ids.empty()) {
        return true;
    }

    std::ifstream in(path_, std::ios::binary);
    if (!in.is_open()) {
        if (err) {
            *err = "BvecsReader::ReadByIds: failed to open file: " + path_;
        }
        return false;
    }

    out->rows = d_;
    out->cols = static_cast<int>(ids.size());
    out->data.resize(static_cast<std::size_t>(d_) * ids.size());

    for (std::size_t j = 0; j < ids.size(); ++j) {
        const std::uint64_t id = ids[j];
        if (id >= n_) {
            if (err) {
                *err = "BvecsReader::ReadByIds: id out of range.";
            }
            return false;
        }
        std::uint8_t* dst = out->Col(static_cast<int>(j));
        const std::uint64_t off = id * rec_;
        if (!ReadRecordAt(in, off, d_, dst, err)) {
            return false;
        }
    }
    return true;
}

bool BvecsReader::ReadOne(std::uint64_t id, std::uint8_t* out_vec, std::string* err) const {
    if (!out_vec) {
        if (err) {
            *err = "BvecsReader::ReadOne: null output.";
        }
        return false;
    }
    if (id >= n_) {
        if (err) {
            *err = "BvecsReader::ReadOne: id out of range.";
        }
        return false;
    }
    std::ifstream in(path_, std::ios::binary);
    if (!in.is_open()) {
        if (err) {
            *err = "BvecsReader::ReadOne: failed to open file: " + path_;
        }
        return false;
    }
    return ReadRecordAt(in, id * rec_, d_, out_vec, err);
}

}  // namespace stlq::io
