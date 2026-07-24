#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace stlq::io {

struct LinkageVirtualStoreConfig {
    std::string dir;
    int nlist = 0;
    int m = 0;        // number of coefficient layers (full m)
    int m_codes = 0;  // stored codes layers (m-1)
};

// Streaming writer for virtual-mode linkage outputs, stored per-cluster in list-order.
// All ids are 0-based global ids for real nodes. Parent is stored as 1-based local index into
// the augmented cluster (real prefix + virtual suffix); 0 means no parent.
class LinkageVirtualWriter {
public:
    bool Open(const LinkageVirtualStoreConfig& cfg, std::string* err);

    // Must be called for every cid in [0, nlist) in increasing order.
    bool AppendCluster(int cid,
                       const std::vector<std::uint32_t>& real_global_ids_by_local, // size n_real (local->global)
                       const std::vector<int>& indices_depth_order,               // size n_real, local ids in depth order
                       const std::vector<std::uint32_t>& parent_local_1based,     // size n_real, values in [0..n_real+n_virt]
                       const std::vector<std::uint32_t>& depth_offsets,           // variable length, last==n_real
                       const std::vector<std::uint8_t>& codes_small_by_local,     // col-major (m_codes×n_real), local order
                       const std::vector<float>& coeffs_by_local,                 // col-major (m×n_real), local order
                       const std::vector<std::uint8_t>& virt_codes_full,          // col-major (m×n_virt)
                       const std::vector<float>& virt_coeffs_full,               // col-major (m×n_virt)
                       int n_real,
                       int n_virt,
                       std::string* err);

    bool Finish(std::string* err) const;

private:
    LinkageVirtualStoreConfig cfg_;

    std::ofstream real_offsets_out_;
    std::ofstream virt_offsets_out_;
    std::ofstream depth_offsets_offsets_out_;
    std::ofstream depth_offsets_out_;
    std::ofstream real_ids_out_;
    std::ofstream parent_out_;
    std::ofstream codes_out_;
    std::ofstream coeffs_out_;
    std::ofstream virt_codes_out_;
    std::ofstream virt_coeffs_out_;

    std::uint64_t real_count_ = 0;
    std::uint64_t virt_count_ = 0;
    std::uint64_t depth_offsets_count_ = 0;
    int next_cid_ = 0;
};

}  // namespace stlq::io

