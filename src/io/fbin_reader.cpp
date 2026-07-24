#include "stlq/io/fbin_reader.h"

#include <algorithm>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace stlq::io {

bool FbinReader::Open(const std::string& path, std::string* err) {
    path_ = path;
    d_ = 0;
    n_ = 0;
    if (path_.empty()) {
        if (err) *err = "FbinReader::Open: empty path.";
        return false;
    }
    std::ifstream in(path_, std::ios::binary);
    if (!in.is_open()) {
        if (err) *err = "FbinReader::Open: failed to open " + path_;
        return false;
    }
    std::int32_t nrows = 0, ncols = 0;
    in.read(reinterpret_cast<char*>(&nrows), sizeof(std::int32_t));
    in.read(reinterpret_cast<char*>(&ncols), sizeof(std::int32_t));
    if (!in || nrows <= 0 || ncols <= 0) {
        if (err) *err = "FbinReader::Open: invalid header in " + path_;
        return false;
    }

    // Verify file size matches header.
    in.seekg(0, std::ios::end);
    const auto size = static_cast<std::uint64_t>(in.tellg());
    const std::uint64_t expected = kHeaderBytes +
        static_cast<std::uint64_t>(nrows) * static_cast<std::uint64_t>(ncols) * sizeof(float);
    if (size < expected) {
        if (err) *err = "FbinReader::Open: file too small (expected " +
                        std::to_string(expected) + " bytes, got " +
                        std::to_string(size) + ") for " + path_;
        return false;
    }

    n_ = static_cast<std::uint64_t>(nrows);
    d_ = static_cast<int>(ncols);
    return true;
}

bool FbinReader::ReadBlock(std::uint64_t start,
                           std::uint32_t count,
                           ColMajorMatrix<float>* out,
                           std::string* err) const {
    if (!out) {
        if (err) *err = "FbinReader::ReadBlock: null output.";
        return false;
    }
    out->rows = d_;
    out->cols = 0;
    out->data.clear();
    if (count == 0 || d_ <= 0 || n_ == 0) {
        return true;
    }
    if (start >= n_) {
        return true;
    }
    const std::uint64_t max_count = n_ - start;
    const std::uint32_t nread = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(max_count, static_cast<std::uint64_t>(count)));

    std::ifstream in(path_, std::ios::binary);
    if (!in.is_open()) {
        if (err) *err = "FbinReader::ReadBlock: failed to open " + path_;
        return false;
    }
    // fbin layout: 8-byte header + row-major float data (no per-row dim prefix).
    const std::uint64_t row_bytes = static_cast<std::uint64_t>(d_) * sizeof(float);
    const std::uint64_t off = kHeaderBytes + start * row_bytes;
    in.seekg(static_cast<std::streamoff>(off), std::ios::beg);
    if (!in) {
        if (err) *err = "FbinReader::ReadBlock: seek failed.";
        return false;
    }

    out->rows = d_;
    out->cols = static_cast<int>(nread);
    out->data.resize(static_cast<std::size_t>(d_) * static_cast<std::size_t>(nread));

    // Read all rows as a contiguous chunk, then transpose row-major → col-major.
    const std::uint64_t bytes = static_cast<std::uint64_t>(nread) * row_bytes;
    scratch_.resize(static_cast<std::size_t>(bytes));
    in.read(reinterpret_cast<char*>(scratch_.data()), static_cast<std::streamsize>(bytes));
    if (!in) {
        if (err) *err = "FbinReader::ReadBlock: read failed.";
        return false;
    }

    for (std::uint32_t i = 0; i < nread; ++i) {
        const auto* src = reinterpret_cast<const float*>(
            scratch_.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(d_) * sizeof(float));
        float* dst = out->Col(static_cast<int>(i));
        std::memcpy(dst, src, sizeof(float) * static_cast<std::size_t>(d_));
    }
    return true;
}

bool FbinReader::ReadBlockInto(std::uint64_t start,
                               std::uint32_t count,
                               float* dst_colmajor,
                               std::size_t dst_bytes,
                               std::string* err) const {
    if (!dst_colmajor) {
        if (err) *err = "FbinReader::ReadBlockInto: null dst.";
        return false;
    }
    if (count == 0 || d_ <= 0 || n_ == 0) {
        return true;
    }
    if (start >= n_) {
        return true;
    }
    const std::uint64_t max_count = n_ - start;
    const std::uint32_t nread = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(max_count, static_cast<std::uint64_t>(count)));
    const std::size_t need = sizeof(float) * static_cast<std::size_t>(d_) * static_cast<std::size_t>(nread);
    if (dst_bytes < need) {
        if (err) *err = "FbinReader::ReadBlockInto: dst_bytes too small.";
        return false;
    }

    std::ifstream in(path_, std::ios::binary);
    if (!in.is_open()) {
        if (err) *err = "FbinReader::ReadBlockInto: failed to open " + path_;
        return false;
    }
    const std::uint64_t row_bytes = static_cast<std::uint64_t>(d_) * sizeof(float);
    const std::uint64_t off = kHeaderBytes + start * row_bytes;
    in.seekg(static_cast<std::streamoff>(off), std::ios::beg);
    if (!in) {
        if (err) *err = "FbinReader::ReadBlockInto: seek failed.";
        return false;
    }

    const std::uint64_t bytes = static_cast<std::uint64_t>(nread) * row_bytes;
    scratch_.resize(static_cast<std::size_t>(bytes));
    in.read(reinterpret_cast<char*>(scratch_.data()), static_cast<std::streamsize>(bytes));
    if (!in) {
        if (err) *err = "FbinReader::ReadBlockInto: read failed.";
        return false;
    }

    for (std::uint32_t i = 0; i < nread; ++i) {
        const auto* src = reinterpret_cast<const float*>(
            scratch_.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(d_) * sizeof(float));
        float* dst = dst_colmajor + static_cast<std::size_t>(i) * static_cast<std::size_t>(d_);
        std::memcpy(dst, src, sizeof(float) * static_cast<std::size_t>(d_));
    }
    return true;
}

bool FbinReader::ReadByIds(const std::vector<std::uint32_t>& ids,
                           ColMajorMatrix<float>* out,
                           std::string* err) const {
    if (!out) {
        if (err) *err = "FbinReader::ReadByIds: null output.";
        return false;
    }
    out->rows = d_;
    out->cols = 0;
    out->data.clear();
    if (ids.empty() || d_ <= 0 || n_ == 0) {
        return true;
    }

    std::ifstream in(path_, std::ios::binary);
    if (!in.is_open()) {
        if (err) *err = "FbinReader::ReadByIds: failed to open " + path_;
        return false;
    }

    out->rows = d_;
    out->cols = static_cast<int>(ids.size());
    out->data.resize(static_cast<std::size_t>(d_) * ids.size());

    const std::uint64_t row_bytes = static_cast<std::uint64_t>(d_) * sizeof(float);

    for (std::size_t j = 0; j < ids.size(); ++j) {
        const auto id = static_cast<std::uint64_t>(ids[j]);
        if (id >= n_) {
            if (err) *err = "FbinReader::ReadByIds: id out of range.";
            return false;
        }
        const std::uint64_t off = kHeaderBytes + id * row_bytes;
        in.seekg(static_cast<std::streamoff>(off), std::ios::beg);
        if (!in) {
            if (err) *err = "FbinReader::ReadByIds: seek failed.";
            return false;
        }
        float* dst = out->Col(static_cast<int>(j));
        in.read(reinterpret_cast<char*>(dst), sizeof(float) * static_cast<std::size_t>(d_));
        if (!in) {
            if (err) *err = "FbinReader::ReadByIds: failed to read vector payload.";
            return false;
        }
    }
    return true;
}

}  // namespace stlq::io
