#include "stlq/io/ibin_reader.h"

#include <algorithm>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace stlq::io {

bool IbinReader::Open(const std::string& path, std::string* err) {
    path_ = path;
    k_ = 0;
    n_ = 0;
    if (path.empty()) {
        if (err) *err = "IbinReader::Open: empty path.";
        return false;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        if (err) *err = "IbinReader::Open: failed to open " + path;
        return false;
    }
    std::int32_t nrows = 0, ncols = 0;
    in.read(reinterpret_cast<char*>(&nrows), sizeof(std::int32_t));
    in.read(reinterpret_cast<char*>(&ncols), sizeof(std::int32_t));
    if (!in || nrows <= 0 || ncols <= 0) {
        if (err) *err = "IbinReader::Open: invalid header in " + path;
        return false;
    }

    in.seekg(0, std::ios::end);
    const auto size = static_cast<std::uint64_t>(in.tellg());
    const std::uint64_t expected = kHeaderBytes +
        static_cast<std::uint64_t>(nrows) * static_cast<std::uint64_t>(ncols) * sizeof(std::int32_t);
    if (size < expected) {
        if (err) *err = "IbinReader::Open: file too small for " + path;
        return false;
    }

    n_ = static_cast<std::uint64_t>(nrows);
    k_ = static_cast<int>(ncols);
    return true;
}

bool IbinReader::ReadBlock(std::uint64_t start,
                           std::uint32_t count,
                           ColMajorMatrix<std::int32_t>* out,
                           std::string* err) const {
    if (!out) {
        if (err) *err = "IbinReader::ReadBlock: null output.";
        return false;
    }
    out->rows = k_;
    out->cols = 0;
    out->data.clear();
    if (count == 0 || k_ <= 0 || n_ == 0) {
        return true;
    }
    if (start >= n_) {
        return true;
    }
    const std::uint64_t avail = n_ - start;
    const std::uint32_t nread = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(avail, static_cast<std::uint64_t>(count)));

    std::ifstream in(path_, std::ios::binary);
    if (!in.is_open()) {
        if (err) *err = "IbinReader::ReadBlock: failed to open " + path_;
        return false;
    }

    const std::uint64_t row_bytes = static_cast<std::uint64_t>(k_) * sizeof(std::int32_t);
    const std::uint64_t off = kHeaderBytes + start * row_bytes;
    in.seekg(static_cast<std::streamoff>(off), std::ios::beg);
    if (!in) {
        if (err) *err = "IbinReader::ReadBlock: seek failed.";
        return false;
    }

    const std::uint64_t bytes = static_cast<std::uint64_t>(nread) * row_bytes;
    scratch_.resize(static_cast<std::size_t>(bytes));
    in.read(reinterpret_cast<char*>(scratch_.data()), static_cast<std::streamsize>(bytes));
    if (!in) {
        if (err) *err = "IbinReader::ReadBlock: read failed.";
        return false;
    }

    out->rows = k_;
    out->cols = static_cast<int>(nread);
    out->data.resize(static_cast<std::size_t>(k_) * static_cast<std::size_t>(nread));

    // Transpose row-major → col-major.
    for (std::uint32_t i = 0; i < nread; ++i) {
        const auto* src = reinterpret_cast<const std::int32_t*>(
            scratch_.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(k_) * sizeof(std::int32_t));
        std::int32_t* dst = out->Col(static_cast<int>(i));
        std::memcpy(dst, src, sizeof(std::int32_t) * static_cast<std::size_t>(k_));
    }
    return true;
}

}  // namespace stlq::io
