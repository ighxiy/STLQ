#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace stlq {

struct RvqLayerStoreConfig {
    std::filesystem::path dir;
    int m = 0;
    std::int64_t n = 0;
};

// -------- Prompt-07: init codes policy (default no-save) --------
enum class RvqCodeDType {
    kU8 = 0,
    kU16 = 1,
    kU32 = 2,
};

inline int RvqCodeBytes(RvqCodeDType t) {
    switch (t) {
    case RvqCodeDType::kU8:
        return 1;
    case RvqCodeDType::kU16:
        return 2;
    case RvqCodeDType::kU32:
        return 4;
    }
    return 0;
}

inline RvqCodeDType RvqCodeDTypeForK(int K) {
    if (K <= 0) return RvqCodeDType::kU8;
    if (K <= 256) return RvqCodeDType::kU8;
    if (K <= static_cast<int>(std::numeric_limits<std::uint16_t>::max()) + 1) return RvqCodeDType::kU16;
    return RvqCodeDType::kU32;
}

struct RvqCodeSpan {
    const void* ptr = nullptr;
    RvqCodeDType dtype = RvqCodeDType::kU8;
};

// Compact per-layer codes kept in host RAM for multi-layer RVQ init.
// This is the default (no disk persistence).
class RvqInitCodesInMemory {
public:
    bool Init(const std::vector<int>& h_vec, std::int64_t n, std::string* err);

    int m() const { return static_cast<int>(layers_.size()); }
    std::int64_t n() const { return n_; }
    RvqCodeDType dtype(int layer) const;
    int bytes_per_code(int layer) const { return RvqCodeBytes(dtype(layer)); }

    // Reset the sequential write cursor for a layer so codes can be re-emitted
    // during the next pass (Prompt-10). This does not clear the underlying buffers.
    bool ResetWriteCursor(int layer, std::string* err);

    // Writes a sequential code block at [col0, col0+cols).
    // `assign_i32` is the argmax assignment (codes in [0,K)).
    bool WriteBlockFromI32(int layer,
                           std::int64_t col0,
                           int cols,
                           const int* assign_i32,
                           std::string* err);

    // Returns a view of codes for [col0, col0+cols). Caller must use dtype() to interpret.
    RvqCodeSpan SpanBlock(int layer, std::int64_t col0, int cols, std::string* err) const;

    // Random-access decode for sampling/validation.
    std::uint32_t CodeAt(int layer, std::int64_t i) const;

private:
    struct Layer {
        RvqCodeDType dtype = RvqCodeDType::kU8;
        std::vector<std::uint8_t> u8;
        std::vector<std::uint16_t> u16;
        std::vector<std::uint32_t> u32;
    };

    std::int64_t n_ = 0;
    std::vector<Layer> layers_;
    std::vector<std::int64_t> next_col0_;
};

struct RvqInitCodesFileConfig {
    std::filesystem::path dir;
    std::vector<int> h_vec;
    std::int64_t n = 0;
};

// Optional debug persistence for init codes (no coefficients).
class RvqInitCodesWriter {
public:
    bool Open(const RvqInitCodesFileConfig& cfg, std::string* err);
    bool WriteBlock(int layer, std::int64_t col0, int cols, const void* codes, RvqCodeDType dtype, std::string* err);
    bool Close(std::string* err);

private:
    RvqInitCodesFileConfig cfg_;
    bool opened_ = false;
    std::vector<RvqCodeDType> dtypes_;
    std::vector<std::int64_t> next_col0_;
    std::vector<std::ofstream> out_;
};

class RvqLayerStoreWriter {
public:
    bool Open(const RvqLayerStoreConfig& cfg, std::string* err);
    bool WriteBlock(int layer,
                    std::int64_t col0,
                    int cols,
                    const std::uint16_t* codes_u16,
                    const float* a_f32,
                    std::string* err);
    // Flush the given layer's streams so they can be read by a reader in the same process.
    // Intended for streaming RVQ where layer l+1 needs to read layer l immediately.
    bool FlushLayer(int layer, std::string* err);
    bool Close(std::string* err);

private:
    RvqLayerStoreConfig cfg_;
    bool opened_ = false;
    std::vector<std::int64_t> next_col0_;
    std::vector<std::ofstream> code_out_;
    std::vector<std::ofstream> a_out_;
};

class RvqLayerStoreReader {
public:
    bool Open(const RvqLayerStoreConfig& cfg, std::string* err);
    bool ResetUpto(int upto_layer, std::string* err);
    bool ReadNextUpto(int upto_layer,
                      int cols,
                      std::vector<std::vector<std::uint16_t>>* codes,
                      std::vector<std::vector<float>>* a,
                      std::string* err);
    bool Close(std::string* err);

private:
    RvqLayerStoreConfig cfg_;
    bool opened_ = false;
    std::vector<std::ifstream> code_in_;
    std::vector<std::ifstream> a_in_;
};

}  // namespace stlq
