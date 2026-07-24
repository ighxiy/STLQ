#include "stlq/io/ivecs_reader.h"

#include <algorithm>
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

}  // namespace

bool IvecsReader::Open(const std::string& path, std::string* err) {
    path_ = path;
    k_ = 0;
    n_ = 0;
    rec_ = 0;

    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        if (err) {
            *err = "IvecsReader: failed to open file: " + path;
        }
        return false;
    }
    std::int32_t k = 0;
    if (!ReadInt32(in, &k)) {
        if (err) {
            *err = "IvecsReader: failed to read dim from: " + path;
        }
        return false;
    }
    if (k <= 0) {
        if (err) {
            *err = "IvecsReader: invalid dim in: " + path;
        }
        return false;
    }

    std::uint64_t size = 0;
    try {
        size = static_cast<std::uint64_t>(std::filesystem::file_size(path));
    } catch (...) {
        if (err) {
            *err = "IvecsReader: failed to stat file: " + path;
        }
        return false;
    }

    const std::uint64_t rec = static_cast<std::uint64_t>(sizeof(std::int32_t)) +
                              static_cast<std::uint64_t>(k) * static_cast<std::uint64_t>(sizeof(std::int32_t));
    const std::uint64_t n = size / rec;
    if (n == 0) {
        if (err) {
            *err = "IvecsReader: empty file: " + path;
        }
        return false;
    }

    k_ = k;
    rec_ = rec;
    n_ = n;
    return true;
}

bool IvecsReader::ReadBlock(std::uint64_t start,
                            std::uint32_t count,
                            ColMajorMatrix<std::int32_t>* out,
                            std::string* err) const {
    if (!out) {
        if (err) {
            *err = "IvecsReader::ReadBlock: null output.";
        }
        return false;
    }
    out->rows = 0;
    out->cols = 0;
    out->data.clear();
    if (k_ <= 0 || rec_ == 0 || n_ == 0) {
        if (err) {
            *err = "IvecsReader::ReadBlock: reader not open.";
        }
        return false;
    }
    if (start >= n_ || count == 0) {
        return true;
    }
    const std::uint64_t avail = n_ - start;
    const std::uint32_t nread = static_cast<std::uint32_t>(std::min<std::uint64_t>(avail, count));

    std::ifstream in(path_, std::ios::binary);
    if (!in.is_open()) {
        if (err) {
            *err = "IvecsReader::ReadBlock: failed to open file: " + path_;
        }
        return false;
    }

    const std::uint64_t offset = start * rec_;
    const std::uint64_t bytes = static_cast<std::uint64_t>(nread) * rec_;
    scratch_.resize(static_cast<std::size_t>(bytes));
    in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!in) {
        if (err) {
            *err = "IvecsReader::ReadBlock: seek failed.";
        }
        return false;
    }
    in.read(reinterpret_cast<char*>(scratch_.data()), static_cast<std::streamsize>(bytes));
    if (!in) {
        if (err) {
            *err = "IvecsReader::ReadBlock: read failed.";
        }
        return false;
    }

    out->rows = k_;
    out->cols = static_cast<int>(nread);
    out->data.resize(static_cast<std::size_t>(k_) * nread);

    for (std::uint32_t i = 0; i < nread; ++i) {
        const std::uint8_t* rec_ptr = scratch_.data() + static_cast<std::size_t>(i) * rec_;
        std::int32_t k = 0;
        std::memcpy(&k, rec_ptr, sizeof(std::int32_t));
        if (k != k_) {
            if (err) {
                *err = "IvecsReader::ReadBlock: dim mismatch (expected " + std::to_string(k_) +
                       ", got " + std::to_string(k) + ")";
            }
            return false;
        }
        std::int32_t* dst = out->Col(static_cast<int>(i));
        std::memcpy(dst, rec_ptr + sizeof(std::int32_t),
                    static_cast<std::size_t>(k_) * sizeof(std::int32_t));
    }
    return true;
}

}  // namespace stlq::io
