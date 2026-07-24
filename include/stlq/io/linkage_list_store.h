#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace stlq::io {

struct LinkageListStoreConfig {
    std::string dir;
    int nlist = 0;
    int m_codes = 0;
    // Small-layer codes (layers 1..m-1): fixed uint8 in the current design.
    // The stored payload is always `m_codes * n_*` bytes.
    static constexpr int kSmallCodeWidthBytes = 1;

    // Layer0 codes for depth>0 nodes (code0_one): only init-linkage legacy stores may use 1/2/4 bytes.
    int code0_width_bytes = 1;
    // If false, do not store float coefficients (coeffs/a0/virt_coeffs/virt_a0) in linkage_list.
    // This is used by the baseset pipeline when coefficients are stored via coeff codec instead.
    bool store_coeffs_f32 = true;
    // If false, do not store `parent.u32`. Parent can still be reconstructed from `parent_louds.bin`
    // if `store_parent_louds=true`. (Some legacy paths may require materializing parent.u32.)
    bool store_parent_u32 = true;
    // Optional parent LOUDS store (additional files; does not remove parent.u32).
    bool store_parent_louds = false;
    int parent_louds_select_stride = 128;
    int parent_louds_rank_words_per_super_log2 = 4;

    // Enable cluster-level checkpoint/resume for the OpenMP baseset linkage stage.
    // NOTE: This is intentionally per-store (not global), because training/iteration-stage
    // linkage_list stores are rebuilt each round and their write plans may change.
    bool enable_checkpoint = false;
};

struct LinkageListMeta {
    int nlist = 0;
    int m_codes = 0;
    int small_code_width_bytes = LinkageListStoreConfig::kSmallCodeWidthBytes;
    int code0_width_bytes = 1;
    int store_coeffs_f32 = 1;
    int store_parent_u32 = 1;
    int store_parent_louds = 0;
    int parent_louds_select_stride = 128;
    int parent_louds_rank_words_per_super_log2 = 4;
};

// Pass-A output: precomputed per-cluster sizes and prefix-sum offsets for the linkage_list store.
// Offsets are counts (not bytes) and always have length (nlist + 1).
struct LinkageListWritePlan {
    int nlist = 0;
    int m_codes = 0;
    int small_code_width_bytes = LinkageListStoreConfig::kSmallCodeWidthBytes;
    int code0_width_bytes = 1;

    std::vector<std::uint64_t> real_offsets;            // counts of real nodes
    std::vector<std::uint64_t> virt_offsets;            // counts of virtual nodes
    std::vector<std::uint64_t> depth_offsets_offsets;   // counts of u32 entries in depth_offsets.u32
    // parent_louds_offsets_bytes: prefix sums of blob byte offsets into parent_louds.bin
    std::vector<std::uint64_t> parent_louds_offsets_bytes;  // bytes, length (nlist+1)

    std::vector<std::uint32_t> n_real;      // per cluster
    std::vector<std::uint32_t> n_virt;      // per cluster
    std::vector<std::uint32_t> depth_len;   // per cluster (u32 count)

    std::uint64_t total_depth_offsets_u32_bytes = 0;
    std::uint64_t total_real_u32_bytes = 0;
    std::uint64_t total_parent_u32_bytes = 0;
    std::uint64_t total_codes_bytes = 0;
    std::uint64_t total_coeffs_f32_bytes = 0;
    std::uint64_t total_code0_one_bytes = 0;
    std::uint64_t total_a0_f32_bytes = 0;
    std::uint64_t total_virt_codes_bytes = 0;
    std::uint64_t total_virt_coeffs_f32_bytes = 0;
    std::uint64_t total_virt_a0_f32_bytes = 0;
    std::uint64_t total_parent_louds_bytes = 0;
};

class LinkageListWriter {
public:
    bool Open(const LinkageListStoreConfig& cfg, std::string* err);

    // Must be called for every cid in [0, nlist) in increasing order.
    // All real arrays are depth-ordered already (no perm stored on disk).
    bool AppendCluster(int cid,
                       const std::vector<std::uint32_t>& real_ids_depth_order,          // size n_real
                       const std::vector<std::uint32_t>& parent_local_1based_depth_order, // size n_real
                       const std::vector<std::uint32_t>& depth_offsets,                // variable, last==n_real
                       const std::vector<std::uint8_t>& codes_small_depth_order_bytes, // m_codes*n_real (uint8)
                       const std::vector<float>& coeffs_small_depth_order,             // m_codes*n_real
                       const std::vector<std::uint8_t>& code0_one_depth_order_bytes,   // n_real*code0_width_bytes (depth>0 only; depth==0 may be 0)
                       const std::vector<float>& a0_depth_order,                        // n_real
                       const std::vector<std::uint8_t>& virt_codes_small_bytes,        // m_codes*n_virt (uint8)
                       const std::vector<float>& virt_coeffs_small,                    // m_codes*n_virt
                       const std::vector<float>& virt_a0,                               // n_virt
                       int n_real,
                       int n_virt,
                       std::string* err);

    bool Finish(std::string* err);

private:
    bool WriteMeta(std::string* err) const;

    LinkageListStoreConfig cfg_;

    std::ofstream real_offsets_out_;
    std::ofstream virt_offsets_out_;
    std::ofstream depth_offsets_offsets_out_;
    std::ofstream depth_offsets_out_;
    std::ofstream real_ids_out_;
    std::ofstream parent_out_;
    std::ofstream parent_louds_offsets_out_;
    std::ofstream parent_louds_out_;
    std::ofstream codes_out_;
    std::ofstream coeffs_out_;
    std::ofstream code0_one_out_;
    std::ofstream a0_out_;
    std::ofstream virt_codes_out_;
    std::ofstream virt_coeffs_out_;
    std::ofstream virt_a0_out_;

    std::uint64_t real_count_ = 0;
    std::uint64_t virt_count_ = 0;
    std::uint64_t depth_offsets_count_ = 0;
    std::uint64_t parent_louds_bytes_ = 0;
    int next_cid_ = 0;
};

// Random (positioned) writer used by the OpenMP per-cluster streaming pipeline.
// `Open()` writes offsets files and preallocates payload files; `WriteClusterAt()` can be called concurrently.
class LinkageListRandomWriter {
public:
    bool Open(const LinkageListStoreConfig& cfg, const LinkageListWritePlan& plan, std::string* err);

    // Resume mode: open an existing partially-written store for positioned writes without truncating
    // the preallocated payload files. This is intended for cluster-level checkpoint/resume.
    //
    // Requirements:
    // - The directory already exists and contains the preallocated payload files of the expected sizes.
    // - The provided plan/config match the existing store layout.
    bool OpenResume(const LinkageListStoreConfig& cfg,
                    const LinkageListWritePlan& plan,
                    std::string* err);

    bool WriteClusterAt(int cid,
                        const LinkageListWritePlan& plan,
                        const std::vector<std::uint32_t>& real_ids_depth_order,
                        const std::vector<std::uint32_t>& parent_local_1based_depth_order,
                        const std::vector<std::uint32_t>& depth_offsets,
                        const std::vector<std::uint8_t>& codes_small_depth_order_bytes,
                        const std::vector<float>& coeffs_small_depth_order,
                        const std::vector<std::uint8_t>& code0_one_depth_order_bytes,
                        const std::vector<float>& a0_depth_order,
                        const std::vector<std::uint8_t>& virt_codes_small_bytes,
                        const std::vector<float>& virt_coeffs_small,
                        const std::vector<float>& virt_a0,
                        std::string* err);

    bool Finish(std::string* err);

private:
    bool WriteMeta(std::string* err) const;

    LinkageListStoreConfig cfg_;

#if defined(_WIN32)
    // Windows fallback: positioned writes via seekp+write guarded by per-file mutexes.
    std::fstream depth_offsets_out_;
    std::fstream real_ids_out_;
    std::fstream parent_out_;
    std::fstream parent_louds_out_;
    std::fstream codes_out_;
    std::fstream coeffs_out_;
    std::fstream code0_one_out_;
    std::fstream a0_out_;
    std::fstream virt_codes_out_;
    std::fstream virt_coeffs_out_;
    std::fstream virt_a0_out_;
#else
    // POSIX (Linux): use pwrite to avoid mutexes.
    int fd_depth_offsets_ = -1;
    int fd_real_ids_ = -1;
    int fd_parent_ = -1;
    int fd_parent_louds_ = -1;
    int fd_codes_ = -1;
    int fd_coeffs_ = -1;
    int fd_code0_one_ = -1;
    int fd_a0_ = -1;
    int fd_virt_codes_ = -1;
    int fd_virt_coeffs_ = -1;
    int fd_virt_a0_ = -1;
#endif
};

class LinkageListReader {
public:
    bool Open(const std::string& dir, std::string* err);

    [[nodiscard]] const std::string& dir() const { return dir_; }
    [[nodiscard]] const LinkageListMeta& meta() const { return meta_; }
    [[nodiscard]] int nlist() const { return meta_.nlist; }
    [[nodiscard]] int m_codes() const { return meta_.m_codes; }
    [[nodiscard]] int small_code_width_bytes() const { return meta_.small_code_width_bytes; }
    [[nodiscard]] int code0_width_bytes() const { return meta_.code0_width_bytes; }
    [[nodiscard]] std::uint64_t total_real() const { return real_offsets_.empty() ? 0 : real_offsets_.back(); }
    [[nodiscard]] std::uint64_t total_virtual() const { return virt_offsets_.empty() ? 0 : virt_offsets_.back(); }
    [[nodiscard]] bool has_parent_louds() const { return meta_.store_parent_louds != 0; }
    [[nodiscard]] int parent_louds_select_stride() const { return meta_.parent_louds_select_stride; }

    bool ReadClusterSpan(int cid,
                         std::uint64_t* real_lo,
                         std::uint64_t* real_hi,
                         std::uint64_t* virt_lo,
                         std::uint64_t* virt_hi,
                         std::uint64_t* depth_lo,
                         std::uint64_t* depth_hi,
                         std::string* err) const;

    bool ReadCluster(int cid,
                     std::vector<std::uint32_t>* real_ids_depth_order,
                     std::vector<std::uint32_t>* parent_local_1based_depth_order,
                     std::vector<std::uint32_t>* depth_offsets,
                     std::vector<std::uint8_t>* codes_small_depth_order_bytes,
                     std::vector<float>* coeffs_small_depth_order,
                     std::vector<std::uint8_t>* code0_one_depth_order_bytes,
                     std::vector<float>* a0_depth_order,
                     std::vector<std::uint8_t>* virt_codes_small_bytes,
                     std::vector<float>* virt_coeffs_small,
                     std::vector<float>* virt_a0,
                     std::string* err) const;

    // Lightweight read for analysis: returns the per-depth prefix offsets for real nodes.
    // This avoids reading large codes/coeffs payloads when you only need linkage depth stats.
    bool ReadClusterDepthOffsets(int cid,
                                 std::vector<std::uint32_t>* depth_offsets,
                                 std::uint32_t* n_real_out,
                                 std::string* err) const;

    bool ReadClusterParentLOUDSBlob(int cid,
                                    std::vector<std::uint8_t>* blob,
                                    std::string* err) const;

    [[nodiscard]] const std::vector<std::uint64_t>& parent_louds_offsets_bytes() const { return parent_louds_offsets_bytes_; }
    [[nodiscard]] const std::vector<std::uint64_t>& real_offsets() const { return real_offsets_; }

private:
    static bool ReadAll(std::ifstream& in, void* dst, std::size_t bytes);
    static bool ReadU32Span(const std::string& path, std::uint64_t begin, std::uint32_t count,
                     std::vector<std::uint32_t>* out, std::string* err) ;
    static bool ReadBytesSpan(const std::string& path, std::uint64_t begin_bytes, std::uint64_t bytes,
                       std::vector<std::uint8_t>* out, std::string* err) ;
    static bool ReadF32Span(const std::string& path, std::uint64_t begin, std::uint32_t count,
                     std::vector<float>* out, std::string* err) ;

    std::string dir_;
    LinkageListMeta meta_;
    std::vector<std::uint64_t> real_offsets_;
    std::vector<std::uint64_t> virt_offsets_;
    std::vector<std::uint64_t> depth_offsets_offsets_;
    std::vector<std::uint64_t> parent_louds_offsets_bytes_;
};

// Thread-local (or single-thread) reader that keeps payload files open to avoid open/close per cluster.
// Not thread-safe for concurrent calls; create one instance per thread.
class LinkageListThreadReader {
public:
    LinkageListThreadReader() = default;
    ~LinkageListThreadReader();

    bool OpenFrom(const LinkageListReader& src, std::string* err);
    bool OpenFrom(const LinkageListReader& src, bool require_coeffs_f32, std::string* err);

    const LinkageListMeta& meta() const { return meta_; }
    int nlist() const { return meta_.nlist; }
    int m_codes() const { return meta_.m_codes; }
    int small_code_width_bytes() const { return meta_.small_code_width_bytes; }
    int code0_width_bytes() const { return meta_.code0_width_bytes; }

    bool ReadCluster(int cid,
                     std::vector<std::uint32_t>* real_ids_depth_order,
                     std::vector<std::uint32_t>* parent_local_1based_depth_order,
                     std::vector<std::uint32_t>* depth_offsets,
                     std::vector<std::uint8_t>* codes_small_depth_order_bytes,
                     std::vector<float>* coeffs_small_depth_order,
                     std::vector<std::uint8_t>* code0_one_depth_order_bytes,
                     std::vector<float>* a0_depth_order,
                     std::vector<std::uint8_t>* virt_codes_small_bytes,
                     std::vector<float>* virt_coeffs_small,
                     std::vector<float>* virt_a0,
                     std::string* err);

    bool ReadClusterParentLOUDSBlob(int cid,
                                    std::vector<std::uint8_t>* blob,
                                    std::string* err);

private:
    void Close();
    bool OpenFiles(std::string* err);
    static bool ReadAt(std::ifstream* in, std::uint64_t off, void* dst, std::size_t bytes, std::string* err);

    const LinkageListReader* src_ = nullptr;
    std::string dir_;
    LinkageListMeta meta_;
    bool require_coeffs_f32_ = true;
    std::vector<std::uint64_t> parent_louds_offsets_bytes_;
    std::ifstream depth_offsets_in_;
    std::ifstream real_ids_in_;
    std::ifstream parent_in_;
    std::ifstream codes_in_;
    std::ifstream coeffs_in_;
    std::ifstream code0_one_in_;
    std::ifstream a0_in_;
    std::ifstream virt_codes_in_;
    std::ifstream virt_coeffs_in_;
    std::ifstream virt_a0_in_;
    std::ifstream parent_louds_in_;
};

// Rewrites linkage_list/meta.bin using the current v4 header format.
bool RewriteLinkageListMeta(const std::string& dir,
                          const LinkageListMeta& meta,
                          std::string* err);

// Convenience helper to toggle the `store_coeffs_f32` flag in linkage_list/meta.bin.
bool UpdateLinkageListMetaStoreCoeffsF32(const std::string& dir,
                                       bool store_coeffs_f32,
                                       std::string* err);

}  // namespace stlq::io
