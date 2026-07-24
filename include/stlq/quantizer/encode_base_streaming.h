#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "stlq/common/config.h"
#include "stlq/common/types.h"

namespace stlq {

class StreamKernelProvider;
struct Precomp;
struct PrecompLargeRoot;

namespace io {
class BvecsReader;
class FvecsReader;
class FbinReader;
class BaseBasicWriter;
}  // namespace io

// Optional hook invoked after each committed output block is written.
// Intended for checkpointing (baseset base_basic stage). Leave nullptr for default behavior.
struct StreamingBasicCommitHook {
    void* ctx = nullptr;
    bool (*OnCommit)(std::uint64_t next_id, io::BaseBasicWriter* writer, void* ctx, std::string* err) = nullptr;
};

// Encode base set out-of-core and write outputs to BaseBasicStore.
// This is the large-scale counterpart of `EncodeBase(...)` (in-mem).
bool EncodeBaseStreaming(const Config& cfg,
                         const io::BvecsReader& base_reader,
                         const ColMajorMatrix<float>& R,   // d×d rotation
                         const CodebookPack& C_root,
                         const Precomp& pre_full,          // for small h0 path
                         const PrecompLargeRoot* pre_large, // optional large-root precomp
                         StreamKernelProvider* kernels,    // optional (CPU default)
                         io::BaseBasicWriter* writer,
                         float* out_beam_mse,
                         float* out_final_mse,
                         std::string* err,
                         std::uint64_t start_id = 0,
                         const StreamingBasicCommitHook* commit_hook = nullptr);

// Streaming base encode for float32 `.fvecs` datasets (e.g., SIFT1M).
// Semantics match `EncodeBaseStreaming` but input vectors are already float32.
bool EncodeBaseStreamingF32(const Config& cfg,
                            const io::FvecsReader& base_reader,
                            const ColMajorMatrix<float>& R,   // d×d rotation
                            const CodebookPack& C_root,
                            const Precomp& pre_full,
                            const PrecompLargeRoot* pre_large,
                            StreamKernelProvider* kernels,    // optional (CPU default)
                            io::BaseBasicWriter* writer,
                            float* out_beam_mse,
                            float* out_final_mse,
                            std::string* err,
                            std::uint64_t start_id = 0,
                            const StreamingBasicCommitHook* commit_hook = nullptr);

// Streaming base encode for float32 `.fbin` datasets (Yandex Deep10M/Deep1B format).
bool EncodeBaseStreamingF32(const Config& cfg,
                            const io::FbinReader& base_reader,
                            const ColMajorMatrix<float>& R,
                            const CodebookPack& C_root,
                            const Precomp& pre_full,
                            const PrecompLargeRoot* pre_large,
                            StreamKernelProvider* kernels,
                            io::BaseBasicWriter* writer,
                            float* out_beam_mse,
                            float* out_final_mse,
                            std::string* err,
                            std::uint64_t start_id = 0,
                            const StreamingBasicCommitHook* commit_hook = nullptr);

// Generic name for the same routine (used for both train-set and base-set streaming basic encoding).
// NOTE: Train-vs-base semantics (e.g. abs vs noabs) are controlled by the caller via cfg fields.
inline bool EncodeBasicStreaming(const Config& cfg,
                                 const io::BvecsReader& reader,
                                 const ColMajorMatrix<float>& R,
                                 const CodebookPack& C_root,
                                 const Precomp& pre_full,
                                 const PrecompLargeRoot* pre_large,
                                 StreamKernelProvider* kernels,
                                 io::BaseBasicWriter* writer,
                                 float* out_beam_mse,
                                 float* out_final_mse,
                                 std::string* err) {
    return EncodeBaseStreaming(cfg, reader, R, C_root, pre_full, pre_large, kernels, writer, out_beam_mse, out_final_mse, err);
}

inline bool EncodeBasicStreamingF32(const Config& cfg,
                                    const io::FvecsReader& reader,
                                    const ColMajorMatrix<float>& R,
                                    const CodebookPack& C_root,
                                    const Precomp& pre_full,
                                    const PrecompLargeRoot* pre_large,
                                    StreamKernelProvider* kernels,
                                    io::BaseBasicWriter* writer,
                                    float* out_beam_mse,
                                    float* out_final_mse,
                                    std::string* err) {
    return EncodeBaseStreamingF32(cfg, reader, R, C_root, pre_full, pre_large, kernels, writer, out_beam_mse, out_final_mse, err);
}

}  // namespace stlq
