#pragma once

#include <atomic>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

namespace stlq::io {

// On-disk storage for lossy (int8+scale) + lossless (Huffman) coefficient codec,
// keyed by IVF cluster id (cid).
//
// Groups:
// - group=0: root (real depth==0)
// - group=1: linkage (real depth>0 + virtual suffix), per the current virtual design.
//
// The store does NOT persist int8 coefficients directly; it persists Huffman code lengths + payloads,
// plus per-(cluster,group,layer) scales.
struct LinkageCoeffCodecStoreConfig {
    std::string dir;
    int nlist = 0;
    int m = 0;
    // "cluster" (1 stream per group) or "layer" (m streams per group).
    std::string granularity = "cluster";
    // Max signed bits across layers (e.g., 7 -> Qmax=63). Used for symbol mapping [-Qmax..Qmax].
    int max_bits = 7;
    // If 256, use full int8 alphabet [-128..127]. Otherwise, use S=2*Qmax+1 (odd).
    int alphabet_size = 0;
};

struct LinkageCoeffCodecMeta {
    int version = 1;
    int nlist = 0;
    int m = 0;
    int streams = 0;
    int max_bits = 0;
    int alphabet_size = 0;
    int S = 0;
};

struct LinkageCoeffCodecStats {
    std::uint64_t lens_bytes = 0;
    std::uint64_t scales_bytes = 0;
    std::uint64_t payload_bytes = 0;
    std::uint64_t payload_offsets_bytes = 0;
    std::uint64_t payload_sizes_bytes = 0;
};

class LinkageCoeffCodecRandomWriter {
public:
    bool Open(const LinkageCoeffCodecStoreConfig& cfg, std::string* err);

    // Resume mode: open an existing partially-written codec store without truncating files.
    // Intended for cluster-level checkpoint/resume.
    bool OpenResume(const LinkageCoeffCodecStoreConfig& cfg, std::string* err);

    // Thread-safe: can be called concurrently for different cids.
    //
    // scales_* size must be m.
    // lens_* is `streams` arrays, each of length S (uint8 lens per symbol).
    // payload_* is `streams` byte buffers.
    bool WriteClusterAt(int cid,
                        const std::vector<float>& scales_root,
                        const std::vector<float>& scales_linkage,
                        const std::vector<std::vector<std::uint8_t>>& lens_root,
                        const std::vector<std::vector<std::uint8_t>>& lens_linkage,
                        const std::vector<std::vector<std::uint8_t>>& payload_root,
                        const std::vector<std::vector<std::uint8_t>>& payload_linkage,
                        std::string* err);

    bool Finish(std::string* err);

    const LinkageCoeffCodecMeta& meta() const { return meta_; }
    const LinkageCoeffCodecStats& stats() const { return stats_; }

private:
    bool WriteMeta(std::string* err);

    LinkageCoeffCodecStoreConfig cfg_;
    LinkageCoeffCodecMeta meta_;
    LinkageCoeffCodecStats stats_;

    std::vector<std::uint64_t> payload_offs_u64_;  // size nlist*2*streams
    std::vector<std::uint32_t> payload_sizes_u32_; // size nlist*2*streams

    std::atomic<std::uint64_t> payload_cursor_{0};

#if defined(_WIN32)
    std::fstream payload_out_;
    std::fstream lens_out_;
    std::fstream scales_out_;
    std::fstream payload_offs_out_;
    std::fstream payload_sizes_out_;
    std::mutex mu_payload_;
    std::mutex mu_lens_;
    std::mutex mu_scales_;
    std::mutex mu_offs_;
#else
    int fd_payload_ = -1;
    int fd_lens_ = -1;
    int fd_scales_ = -1;
    int fd_payload_offs_ = -1;
    int fd_payload_sizes_ = -1;
#endif
};

class LinkageCoeffCodecReader {
public:
    bool Open(const std::string& dir, std::string* err);

    const LinkageCoeffCodecMeta& meta() const { return meta_; }

    // Read per-cluster scales and Huffman lens + payloads.
    bool ReadCluster(int cid,
                     std::vector<float>* scales_root,
                     std::vector<float>* scales_linkage,
                     std::vector<std::vector<std::uint8_t>>* lens_root,
                     std::vector<std::vector<std::uint8_t>>* lens_linkage,
                     std::vector<std::vector<std::uint8_t>>* payload_root,
                     std::vector<std::vector<std::uint8_t>>* payload_linkage,
                     std::string* err) const;

    const std::string& dir() const { return dir_; }
    const std::vector<std::uint64_t>& payload_offs_u64() const { return payload_offs_u64_; }
    const std::vector<std::uint32_t>& payload_sizes_u32() const { return payload_sizes_u32_; }

private:
    LinkageCoeffCodecMeta meta_;
    std::string dir_;
    std::vector<std::uint64_t> payload_offs_u64_;
    std::vector<std::uint32_t> payload_sizes_u32_;
};

class LinkageCoeffCodecThreadReader {
public:
    LinkageCoeffCodecThreadReader() = default;
    ~LinkageCoeffCodecThreadReader();

    bool OpenFrom(const LinkageCoeffCodecReader& src, std::string* err);

    const LinkageCoeffCodecMeta& meta() const { return meta_; }

    bool ReadCluster(int cid,
                     std::vector<float>* scales_root,
                     std::vector<float>* scales_linkage,
                     std::vector<std::vector<std::uint8_t>>* lens_root,
                     std::vector<std::vector<std::uint8_t>>* lens_linkage,
                     std::vector<std::vector<std::uint8_t>>* payload_root,
                     std::vector<std::vector<std::uint8_t>>* payload_linkage,
                     std::string* err);

private:
    void Close();
    bool OpenFiles(std::string* err);

    static bool ReadAt(std::ifstream* in, std::uint64_t off, void* dst, std::size_t bytes, std::string* err);

    const LinkageCoeffCodecReader* src_ = nullptr;
    std::string dir_;
    LinkageCoeffCodecMeta meta_;

    std::ifstream scales_in_;
    std::ifstream lens_in_;
    std::ifstream payload_in_;
};

}  // namespace stlq::io
