#include "stlq/quantizer/rvq_layer_store.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>
#include <type_traits>

namespace stlq {

namespace {

std::filesystem::path LayerCodePath(const std::filesystem::path& dir, int layer) {
    std::ostringstream oss;
    oss << "rvq_layer_" << std::setw(2) << std::setfill('0') << layer << "_code.u16";
    return dir / oss.str();
}

std::filesystem::path LayerAPath(const std::filesystem::path& dir, int layer) {
    std::ostringstream oss;
    oss << "rvq_layer_" << std::setw(2) << std::setfill('0') << layer << "_a.f32";
    return dir / oss.str();
}

std::filesystem::path LayerCodePathTyped(const std::filesystem::path& dir, int layer, RvqCodeDType dtype) {
    std::ostringstream oss;
    const char* ext = "u8";
    switch (dtype) {
    case RvqCodeDType::kU16:
        ext = "u16";
        break;
    case RvqCodeDType::kU32:
        ext = "u32";
        break;
    default:
        break;
    }
    oss << "rvq_layer_" << std::setw(2) << std::setfill('0') << layer << "_code." << ext;
    return dir / oss.str();
}

template <typename T>
bool ReadExact(std::ifstream& in, T* dst, std::size_t n, std::string* err) {
    in.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(n * sizeof(T)));
    if (in.gcount() != static_cast<std::streamsize>(n * sizeof(T))) {
        if (err) *err = "RvqLayerStoreReader: short read.";
        return false;
    }
    return true;
}

}  // namespace

RvqCodeDType RvqInitCodesInMemory::dtype(int layer) const {
    if (layer < 0 || layer >= static_cast<int>(layers_.size())) return RvqCodeDType::kU8;
    return layers_[static_cast<std::size_t>(layer)].dtype;
}

bool RvqInitCodesInMemory::ResetWriteCursor(int layer, std::string* err) {
    if (layer < 0 || layer >= static_cast<int>(layers_.size())) {
        if (err) *err = "RvqInitCodesInMemory::ResetWriteCursor: layer out of range.";
        return false;
    }
    next_col0_[static_cast<std::size_t>(layer)] = 0;
    return true;
}

bool RvqInitCodesInMemory::Init(const std::vector<int>& h_vec, std::int64_t n, std::string* err) {
    if (h_vec.empty() || n <= 0) {
        if (err) *err = "RvqInitCodesInMemory::Init: invalid h_vec/n.";
        return false;
    }
    // NOTE: On 64-bit platforms, `size_t` may be unsigned 64-bit while `int64_t` is signed.
    // Casting `size_t::max()` to `int64_t` can wrap to -1 and break this check.
    // Compare in unsigned space after rejecting negatives.
    const auto n_u64 = static_cast<std::uint64_t>(n);
    const auto size_t_max_u64 = static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());
    if (n_u64 > size_t_max_u64) {
        if (err) {
            *err = "RvqInitCodesInMemory::Init: n exceeds size_t (n=" + std::to_string(n_u64) +
                   " size_t_max=" + std::to_string(size_t_max_u64) + ").";
        }
        return false;
    }

    n_ = n;
    layers_.assign(h_vec.size(), {});
    next_col0_.assign(h_vec.size(), 0);
    try {
        for (std::size_t l = 0; l < h_vec.size(); ++l) {
            Layer& layer = layers_[l];
            layer.dtype = RvqCodeDTypeForK(h_vec[l]);
            const auto nn = static_cast<std::size_t>(n);
            if (layer.dtype == RvqCodeDType::kU8) {
                layer.u8.assign(nn, 0);
            } else if (layer.dtype == RvqCodeDType::kU16) {
                layer.u16.assign(nn, 0);
            } else {
                layer.u32.assign(nn, 0);
            }
        }
    } catch (const std::bad_alloc&) {
        if (err) *err = "RvqInitCodesInMemory::Init: OOM allocating codes.";
        layers_.clear();
        next_col0_.clear();
        n_ = 0;
        return false;
    }
    return true;
}

bool RvqInitCodesInMemory::WriteBlockFromI32(int layer,
                                            std::int64_t col0,
                                            int cols,
                                            const int* assign_i32,
                                            std::string* err) {
    if (layer < 0 || layer >= static_cast<int>(layers_.size())) {
        if (err) *err = "RvqInitCodesInMemory::WriteBlockFromI32: layer out of range.";
        return false;
    }
    if (cols <= 0 || col0 < 0 || col0 + cols > n_) {
        if (err) *err = "RvqInitCodesInMemory::WriteBlockFromI32: invalid col0/cols.";
        return false;
    }
    if (!assign_i32) {
        if (err) *err = "RvqInitCodesInMemory::WriteBlockFromI32: assign_i32 is null.";
        return false;
    }
    const auto li = static_cast<std::size_t>(layer);
    if (col0 != next_col0_[li]) {
        if (err) *err = "RvqInitCodesInMemory::WriteBlockFromI32: non-sequential write (col0 mismatch).";
        return false;
    }

    Layer& dst = layers_[li];
    const auto off = static_cast<std::size_t>(col0);
    if (dst.dtype == RvqCodeDType::kU8) {
        for (int j = 0; j < cols; ++j) {
            const int v = assign_i32[j];
            dst.u8[off + static_cast<std::size_t>(j)] = static_cast<std::uint8_t>(v);
        }
    } else if (dst.dtype == RvqCodeDType::kU16) {
        for (int j = 0; j < cols; ++j) {
            const int v = assign_i32[j];
            dst.u16[off + static_cast<std::size_t>(j)] = static_cast<std::uint16_t>(v);
        }
    } else {
        for (int j = 0; j < cols; ++j) {
            const int v = assign_i32[j];
            dst.u32[off + static_cast<std::size_t>(j)] = static_cast<std::uint32_t>(v);
        }
    }

    next_col0_[li] += cols;
    return true;
}

RvqCodeSpan RvqInitCodesInMemory::SpanBlock(int layer, std::int64_t col0, int cols, std::string* err) const {
    if (layer < 0 || layer >= static_cast<int>(layers_.size())) {
        if (err) *err = "RvqInitCodesInMemory::SpanBlock: layer out of range.";
        return {};
    }
    if (cols <= 0 || col0 < 0 || col0 + cols > n_) {
        if (err) *err = "RvqInitCodesInMemory::SpanBlock: invalid col0/cols.";
        return {};
    }
    const Layer& src = layers_[static_cast<std::size_t>(layer)];
    const auto off = static_cast<std::size_t>(col0);
    if (src.dtype == RvqCodeDType::kU8) {
        return {src.u8.data() + off, src.dtype};
    }
    if (src.dtype == RvqCodeDType::kU16) {
        return {src.u16.data() + off, src.dtype};
    }
    return {src.u32.data() + off, src.dtype};
}

std::uint32_t RvqInitCodesInMemory::CodeAt(int layer, std::int64_t i) const {
    // Residual-init callers validate upto_layer <= m() and prior_codes.n == reader.n;
    // sample ids come from reader block ranges.
    // if (layer < 0 || layer >= static_cast<int>(layers_.size())) return 0;
    // if (i < 0 || i >= n_) return 0;
    const Layer& src = layers_[static_cast<std::size_t>(layer)];
    const auto idx = static_cast<std::size_t>(i);
    if (src.dtype == RvqCodeDType::kU8) return static_cast<std::uint32_t>(src.u8[idx]);
    if (src.dtype == RvqCodeDType::kU16) return static_cast<std::uint32_t>(src.u16[idx]);
    return src.u32[idx];
}

bool RvqInitCodesWriter::Open(const RvqInitCodesFileConfig& cfg, std::string* err) {
    if (cfg.h_vec.empty() || cfg.n <= 0) {
        if (err) *err = "RvqInitCodesWriter::Open: invalid h_vec/n.";
        return false;
    }
    cfg_ = cfg;
    std::error_code ec;
    std::filesystem::create_directories(cfg_.dir, ec);
    if (ec) {
        if (err) *err = "RvqInitCodesWriter::Open: create_directories failed: " + ec.message();
        return false;
    }
    opened_ = true;
    const int m = static_cast<int>(cfg_.h_vec.size());
    dtypes_.assign(static_cast<std::size_t>(m), RvqCodeDType::kU8);
    next_col0_.assign(static_cast<std::size_t>(m), 0);
    out_.clear();
    out_.reserve(static_cast<std::size_t>(m));
    for (int l = 0; l < m; ++l) {
        const RvqCodeDType dt = RvqCodeDTypeForK(cfg_.h_vec[static_cast<std::size_t>(l)]);
        dtypes_[static_cast<std::size_t>(l)] = dt;
        std::ofstream f(LayerCodePathTyped(cfg_.dir, l, dt), std::ios::binary | std::ios::trunc);
        if (!f) {
            if (err) *err = "RvqInitCodesWriter::Open: failed to open code file.";
            return false;
        }
        out_.push_back(std::move(f));
    }
    return true;
}

bool RvqInitCodesWriter::WriteBlock(int layer,
                                    std::int64_t col0,
                                    int cols,
                                    const void* codes,
                                    RvqCodeDType dtype,
                                    std::string* err) {
    if (!opened_) {
        if (err) *err = "RvqInitCodesWriter::WriteBlock: not opened.";
        return false;
    }
    if (layer < 0 || layer >= static_cast<int>(cfg_.h_vec.size())) {
        if (err) *err = "RvqInitCodesWriter::WriteBlock: layer out of range.";
        return false;
    }
    if (cols <= 0 || col0 < 0 || col0 + cols > cfg_.n) {
        if (err) *err = "RvqInitCodesWriter::WriteBlock: invalid col0/cols.";
        return false;
    }
    if (!codes) {
        if (err) *err = "RvqInitCodesWriter::WriteBlock: codes is null.";
        return false;
    }
    const auto li = static_cast<std::size_t>(layer);
    if (dtype != dtypes_[li]) {
        if (err) *err = "RvqInitCodesWriter::WriteBlock: dtype mismatch.";
        return false;
    }
    if (col0 != next_col0_[li]) {
        if (err) *err = "RvqInitCodesWriter::WriteBlock: non-sequential write (col0 mismatch).";
        return false;
    }
    const std::size_t bytes = static_cast<std::size_t>(cols) * static_cast<std::size_t>(RvqCodeBytes(dtype));
    out_[li].write(reinterpret_cast<const char*>(codes), static_cast<std::streamsize>(bytes));
    if (!out_[li]) {
        if (err) *err = "RvqInitCodesWriter::WriteBlock: write failed.";
        return false;
    }
    next_col0_[li] += cols;
    return true;
}

bool RvqInitCodesWriter::Close(std::string* err) {
    if (!opened_) return true;
    for (int l = 0; l < static_cast<int>(cfg_.h_vec.size()); ++l) {
        const auto li = static_cast<std::size_t>(l);
        if (next_col0_[li] != cfg_.n) {
            if (err) *err = "RvqInitCodesWriter::Close: layer not fully written.";
            return false;
        }
    }
    out_.clear();
    dtypes_.clear();
    next_col0_.clear();
    opened_ = false;
    return true;
}

bool RvqLayerStoreWriter::Open(const RvqLayerStoreConfig& cfg, std::string* err) {
    if (cfg.m <= 0 || cfg.n <= 0) {
        if (err) *err = "RvqLayerStoreWriter::Open: invalid m/n.";
        return false;
    }
    cfg_ = cfg;
    std::error_code ec;
    std::filesystem::create_directories(cfg_.dir, ec);
    if (ec) {
        if (err) *err = "RvqLayerStoreWriter::Open: create_directories failed: " + ec.message();
        return false;
    }
    next_col0_.assign(static_cast<std::size_t>(cfg_.m), 0);
    code_out_.clear();
    a_out_.clear();
    code_out_.reserve(static_cast<std::size_t>(cfg_.m));
    a_out_.reserve(static_cast<std::size_t>(cfg_.m));
    for (int l = 0; l < cfg_.m; ++l) {
        std::ofstream code(LayerCodePath(cfg_.dir, l), std::ios::binary | std::ios::trunc);
        if (!code) {
            if (err) *err = "RvqLayerStoreWriter::Open: failed to open code file.";
            return false;
        }
        std::ofstream a(LayerAPath(cfg_.dir, l), std::ios::binary | std::ios::trunc);
        if (!a) {
            if (err) *err = "RvqLayerStoreWriter::Open: failed to open a file.";
            return false;
        }
        code_out_.push_back(std::move(code));
        a_out_.push_back(std::move(a));
    }
    opened_ = true;
    return true;
}

bool RvqLayerStoreWriter::WriteBlock(int layer,
                                     std::int64_t col0,
                                     int cols,
                                     const std::uint16_t* codes_u16,
                                     const float* a_f32,
                                     std::string* err) {
    if (!opened_) {
        if (err) *err = "RvqLayerStoreWriter::WriteBlock: not opened.";
        return false;
    }
    if (layer < 0 || layer >= cfg_.m) {
        if (err) *err = "RvqLayerStoreWriter::WriteBlock: layer out of range.";
        return false;
    }
    if (cols <= 0 || col0 < 0 || col0 + cols > cfg_.n) {
        if (err) *err = "RvqLayerStoreWriter::WriteBlock: invalid col0/cols.";
        return false;
    }
    if (!codes_u16 || !a_f32) {
        if (err) *err = "RvqLayerStoreWriter::WriteBlock: null pointers.";
        return false;
    }
    if (col0 != next_col0_[static_cast<std::size_t>(layer)]) {
        if (err) *err = "RvqLayerStoreWriter::WriteBlock: non-sequential write (col0 mismatch).";
        return false;
    }
    std::ofstream& code = code_out_[static_cast<std::size_t>(layer)];
    std::ofstream& a = a_out_[static_cast<std::size_t>(layer)];
    code.write(reinterpret_cast<const char*>(codes_u16),
               static_cast<std::streamsize>(sizeof(std::uint16_t) * static_cast<std::size_t>(cols)));
    a.write(reinterpret_cast<const char*>(a_f32),
            static_cast<std::streamsize>(sizeof(float) * static_cast<std::size_t>(cols)));
    if (!code || !a) {
        if (err) *err = "RvqLayerStoreWriter::WriteBlock: write failed.";
        return false;
    }
    next_col0_[static_cast<std::size_t>(layer)] += cols;
    return true;
}

bool RvqLayerStoreWriter::FlushLayer(int layer, std::string* err) {
    if (!opened_) {
        if (err) *err = "RvqLayerStoreWriter::FlushLayer: not opened.";
        return false;
    }
    if (layer < 0 || layer >= cfg_.m) {
        if (err) *err = "RvqLayerStoreWriter::FlushLayer: layer out of range.";
        return false;
    }
    auto& code = code_out_[static_cast<std::size_t>(layer)];
    auto& a = a_out_[static_cast<std::size_t>(layer)];
    code.flush();
    a.flush();
    if (!code || !a) {
        if (err) *err = "RvqLayerStoreWriter::FlushLayer: flush failed.";
        return false;
    }
    return true;
}

bool RvqLayerStoreWriter::Close(std::string* err) {
    if (!opened_) return true;
    for (int l = 0; l < cfg_.m; ++l) {
        const std::int64_t written = next_col0_[static_cast<std::size_t>(l)];
        if (written != cfg_.n) {
            if (err) *err = "RvqLayerStoreWriter::Close: layer not fully written.";
            return false;
        }
    }
    code_out_.clear();
    a_out_.clear();
    next_col0_.clear();
    opened_ = false;
    return true;
}

bool RvqLayerStoreReader::Open(const RvqLayerStoreConfig& cfg, std::string* err) {
    if (cfg.m <= 0 || cfg.n <= 0) {
        if (err) *err = "RvqLayerStoreReader::Open: invalid m/n.";
        return false;
    }
    cfg_ = cfg;
    code_in_.clear();
    a_in_.clear();
    code_in_.reserve(static_cast<std::size_t>(cfg_.m));
    a_in_.reserve(static_cast<std::size_t>(cfg_.m));
    for (int l = 0; l < cfg_.m; ++l) {
        std::ifstream code(LayerCodePath(cfg_.dir, l), std::ios::binary);
        if (!code) {
            if (err) *err = "RvqLayerStoreReader::Open: failed to open code file.";
            return false;
        }
        std::ifstream a(LayerAPath(cfg_.dir, l), std::ios::binary);
        if (!a) {
            if (err) *err = "RvqLayerStoreReader::Open: failed to open a file.";
            return false;
        }
        code_in_.push_back(std::move(code));
        a_in_.push_back(std::move(a));
    }
    opened_ = true;
    return true;
}

bool RvqLayerStoreReader::ResetUpto(int upto_layer, std::string* err) {
    if (!opened_) {
        if (err) *err = "RvqLayerStoreReader::ResetUpto: not opened.";
        return false;
    }
    upto_layer = std::min(std::max(0, upto_layer), cfg_.m);
    for (int l = 0; l < upto_layer; ++l) {
        auto& f0 = code_in_[static_cast<std::size_t>(l)];
        auto& f1 = a_in_[static_cast<std::size_t>(l)];
        f0.clear();
        f1.clear();
        f0.seekg(0, std::ios::beg);
        f1.seekg(0, std::ios::beg);
        if (!f0 || !f1) {
            if (err) *err = "RvqLayerStoreReader::ResetUpto: seek failed.";
            return false;
        }
    }
    return true;
}

bool RvqLayerStoreReader::ReadNextUpto(int upto_layer,
                                       int cols,
                                       std::vector<std::vector<std::uint16_t>>* codes,
                                       std::vector<std::vector<float>>* a,
                                       std::string* err) {
    if (!opened_) {
        if (err) *err = "RvqLayerStoreReader::ReadNextUpto: not opened.";
        return false;
    }
    if (!codes || !a) {
        if (err) *err = "RvqLayerStoreReader::ReadNextUpto: outputs are null.";
        return false;
    }
    upto_layer = std::min(std::max(0, upto_layer), cfg_.m);
    cols = std::max(0, cols);
    codes->assign(static_cast<std::size_t>(upto_layer), {});
    a->assign(static_cast<std::size_t>(upto_layer), {});
    for (int l = 0; l < upto_layer; ++l) {
        auto& out_c = (*codes)[static_cast<std::size_t>(l)];
        auto& out_a = (*a)[static_cast<std::size_t>(l)];
        out_c.resize(static_cast<std::size_t>(cols));
        out_a.resize(static_cast<std::size_t>(cols));
        if (!ReadExact(code_in_[static_cast<std::size_t>(l)], out_c.data(),
                       static_cast<std::size_t>(cols), err)) {
            return false;
        }
        if (!ReadExact(a_in_[static_cast<std::size_t>(l)], out_a.data(),
                       static_cast<std::size_t>(cols), err)) {
            return false;
        }
    }
    return true;
}

bool RvqLayerStoreReader::Close(std::string* /*err*/) {
    code_in_.clear();
    a_in_.clear();
    opened_ = false;
    return true;
}

}  // namespace stlq
