#include "stlq/io/fvecs_reader.h"

#include <cstring>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace stlq::io {

namespace {

bool ReadI32(std::ifstream& in, std::int32_t* out) {
    in.read(reinterpret_cast<char*>(out), sizeof(std::int32_t));
    return static_cast<std::size_t>(in.gcount()) == sizeof(std::int32_t);
}

}  // namespace

bool FvecsReader::Open(const std::string& path, std::string* err) {
    path_ = path;
    d_ = 0;
    n_ = 0;
    rec_ = 0;
    if (path_.empty()) {
        if (err) *err = "FvecsReader::Open: empty path.";
        return false;
    }
    std::ifstream in(path_, std::ios::binary);
    if (!in.is_open()) {
        if (err) *err = "FvecsReader::Open: failed to open " + path_;
        return false;
    }
    std::int32_t d = 0;
    if (!ReadI32(in, &d) || d <= 0) {
        if (err) *err = "FvecsReader::Open: invalid dimension header.";
        return false;
    }
    in.seekg(0, std::ios::end);
    const std::streamoff size = in.tellg();
    rec_ = static_cast<std::uint64_t>(sizeof(std::int32_t)) + static_cast<std::uint64_t>(d) * sizeof(float);
    if (rec_ == 0 || size < 0) {
        if (err) *err = "FvecsReader::Open: invalid file size.";
        return false;
    }
    n_ = static_cast<std::uint64_t>(size) / rec_;
    d_ = static_cast<int>(d);
    return true;
}

bool FvecsReader::ReadBlock(std::uint64_t start,
                            std::uint32_t count,
                            ColMajorMatrix<float>* out,
                            std::string* err) const {
    if (!out) {
        if (err) *err = "FvecsReader::ReadBlock: null output.";
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
        if (err) *err = "FvecsReader::ReadBlock: failed to open " + path_;
        return false;
    }
    const std::uint64_t off = start * rec_;
    in.seekg(static_cast<std::streamoff>(off), std::ios::beg);
    if (!in) {
        if (err) *err = "FvecsReader::ReadBlock: seek failed.";
        return false;
    }

    out->rows = d_;
    out->cols = static_cast<int>(nread);
    out->data.resize(static_cast<std::size_t>(d_) * static_cast<std::size_t>(nread));

    // Bulk-read the raw bytes and unpack to skip per-record dimension headers.
    const std::uint64_t bytes = static_cast<std::uint64_t>(nread) * rec_;
    scratch_.resize(static_cast<std::size_t>(bytes));
    in.read(reinterpret_cast<char*>(scratch_.data()), static_cast<std::streamsize>(bytes));
    if (!in) {
        if (err) *err = "FvecsReader::ReadBlock: read failed.";
        return false;
    }

    for (std::uint32_t i = 0; i < nread; ++i) {
        const std::uint8_t* rec_ptr = scratch_.data() + static_cast<std::size_t>(i) * rec_;
        std::int32_t dim = 0;
        std::memcpy(&dim, rec_ptr, sizeof(std::int32_t));
        if (dim != d_) {
            if (err) *err = "FvecsReader::ReadBlock: dimension mismatch.";
            return false;
        }
        float* dst = out->Col(static_cast<int>(i));
        std::memcpy(dst,
                    rec_ptr + sizeof(std::int32_t),
                    sizeof(float) * static_cast<std::size_t>(d_));
    }
    return true;
}

bool FvecsReader::ReadBlockInto(std::uint64_t start,
                                std::uint32_t count,
                                float* dst_colmajor,
                                std::size_t dst_bytes,
                                std::string* err) const {
    if (!dst_colmajor) {
        if (err) *err = "FvecsReader::ReadBlockInto: null dst.";
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
        if (err) *err = "FvecsReader::ReadBlockInto: dst_bytes too small.";
        return false;
    }

    std::ifstream in(path_, std::ios::binary);
    if (!in.is_open()) {
        if (err) *err = "FvecsReader::ReadBlockInto: failed to open " + path_;
        return false;
    }
    const std::uint64_t off = start * rec_;
    in.seekg(static_cast<std::streamoff>(off), std::ios::beg);
    if (!in) {
        if (err) *err = "FvecsReader::ReadBlockInto: seek failed.";
        return false;
    }

    const std::uint64_t bytes = static_cast<std::uint64_t>(nread) * rec_;
    scratch_.resize(static_cast<std::size_t>(bytes));
    in.read(reinterpret_cast<char*>(scratch_.data()), static_cast<std::streamsize>(bytes));
    if (!in) {
        if (err) *err = "FvecsReader::ReadBlockInto: read failed.";
        return false;
    }

    for (std::uint32_t i = 0; i < nread; ++i) {
        const std::uint8_t* rec_ptr = scratch_.data() + static_cast<std::size_t>(i) * rec_;
        std::int32_t dim = 0;
        std::memcpy(&dim, rec_ptr, sizeof(std::int32_t));
        if (dim != d_) {
            if (err) *err = "FvecsReader::ReadBlockInto: dimension mismatch.";
            return false;
        }
        float* dst = dst_colmajor + static_cast<std::size_t>(i) * static_cast<std::size_t>(d_);
        std::memcpy(dst,
                    rec_ptr + sizeof(std::int32_t),
                    sizeof(float) * static_cast<std::size_t>(d_));
    }
    return true;
}

bool FvecsReader::ReadByIds(const std::vector<std::uint32_t>& ids,
                            ColMajorMatrix<float>* out,
                            std::string* err) const {
    if (!out) {
        if (err) *err = "FvecsReader::ReadByIds: null output.";
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
        if (err) *err = "FvecsReader::ReadByIds: failed to open " + path_;
        return false;
    }

    out->rows = d_;
    out->cols = static_cast<int>(ids.size());
    out->data.resize(static_cast<std::size_t>(d_) * ids.size());

    for (std::size_t j = 0; j < ids.size(); ++j) {
        const auto id = static_cast<std::uint64_t>(ids[j]);
        if (id >= n_) {
            if (err) *err = "FvecsReader::ReadByIds: id out of range.";
            return false;
        }
        const std::uint64_t off = id * rec_;
        in.seekg(static_cast<std::streamoff>(off), std::ios::beg);
        if (!in) {
            if (err) *err = "FvecsReader::ReadByIds: seek failed.";
            return false;
        }

        std::int32_t drec = 0;
        in.read(reinterpret_cast<char*>(&drec), sizeof(std::int32_t));
        if (!in || drec != d_) {
            if (err) *err = "FvecsReader::ReadByIds: dimension mismatch.";
            return false;
        }
        float* dst = out->Col(static_cast<int>(j));
        in.read(reinterpret_cast<char*>(dst), sizeof(float) * static_cast<std::size_t>(d_));
        if (!in) {
            if (err) *err = "FvecsReader::ReadByIds: failed to read vector payload.";
            return false;
        }
    }
    return true;
}

}  // namespace stlq::io
