#include "stlq/io/linkage_store_virtual.h"

#include <filesystem>

namespace stlq::io {

namespace {

bool EnsureDir(const std::string& dir, std::string* err) {
    try {
        std::filesystem::create_directories(dir);
        return true;
    } catch (...) {
        if (err) {
            *err = "LinkageVirtualWriter: failed to create dir: " + dir;
        }
        return false;
    }
}

std::string JoinPath(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    if (a.back() == '/' || a.back() == '\\') return a + b;
    return a + "/" + b;
}

template <typename T>
bool WriteAll(std::ofstream* out, const T* ptr, std::size_t n, std::string* err) {
    if (n == 0) {
        return true;
    }
    out->write(reinterpret_cast<const char*>(ptr), static_cast<std::streamsize>(n * sizeof(T)));
    if (!(*out)) {
        if (err) *err = "LinkageVirtualWriter: write failed.";
        return false;
    }
    return true;
}

}  // namespace

bool LinkageVirtualWriter::Open(const LinkageVirtualStoreConfig& cfg, std::string* err) {
    cfg_ = cfg;
    real_count_ = 0;
    virt_count_ = 0;
    depth_offsets_count_ = 0;
    next_cid_ = 0;

    if (cfg_.dir.empty() || cfg_.nlist <= 0 || cfg_.m <= 0 || cfg_.m_codes < 0) {
        if (err) *err = "LinkageVirtualWriter::Open: invalid config.";
        return false;
    }
    if (!EnsureDir(cfg_.dir, err)) {
        return false;
    }

    real_offsets_out_.open(JoinPath(cfg_.dir, "real_offsets.u64"), std::ios::binary | std::ios::trunc);
    virt_offsets_out_.open(JoinPath(cfg_.dir, "virt_offsets.u64"), std::ios::binary | std::ios::trunc);
    depth_offsets_offsets_out_.open(JoinPath(cfg_.dir, "depth_offsets_offsets.u64"), std::ios::binary | std::ios::trunc);
    depth_offsets_out_.open(JoinPath(cfg_.dir, "depth_offsets.u32"), std::ios::binary | std::ios::trunc);
    real_ids_out_.open(JoinPath(cfg_.dir, "real_ids.u32"), std::ios::binary | std::ios::trunc);
    parent_out_.open(JoinPath(cfg_.dir, "parent_local.u32"), std::ios::binary | std::ios::trunc);
    codes_out_.open(JoinPath(cfg_.dir, "codes_small.u8"), std::ios::binary | std::ios::trunc);
    coeffs_out_.open(JoinPath(cfg_.dir, "coeffs.f32"), std::ios::binary | std::ios::trunc);
    virt_codes_out_.open(JoinPath(cfg_.dir, "virt_codes.u8"), std::ios::binary | std::ios::trunc);
    virt_coeffs_out_.open(JoinPath(cfg_.dir, "virt_coeffs.f32"), std::ios::binary | std::ios::trunc);

    if (!real_offsets_out_.is_open() || !virt_offsets_out_.is_open() || !depth_offsets_offsets_out_.is_open() ||
        !depth_offsets_out_.is_open() || !real_ids_out_.is_open() || !parent_out_.is_open() ||
        !codes_out_.is_open() || !coeffs_out_.is_open() || !virt_codes_out_.is_open() || !virt_coeffs_out_.is_open()) {
        if (err) *err = "LinkageVirtualWriter::Open: failed to open output files.";
        return false;
    }

    // Initial prefix sums.
    const std::uint64_t z = 0;
    real_offsets_out_.write(reinterpret_cast<const char*>(&z), sizeof(z));
    virt_offsets_out_.write(reinterpret_cast<const char*>(&z), sizeof(z));
    depth_offsets_offsets_out_.write(reinterpret_cast<const char*>(&z), sizeof(z));
    if (!real_offsets_out_ || !virt_offsets_out_ || !depth_offsets_offsets_out_) {
        if (err) *err = "LinkageVirtualWriter::Open: failed to write initial offsets.";
        return false;
    }

    return true;
}

bool LinkageVirtualWriter::AppendCluster(int cid,
                                      const std::vector<std::uint32_t>& real_global_ids_by_local,
                                      const std::vector<int>& indices_depth_order,
                                      const std::vector<std::uint32_t>& parent_local_1based,
                                      const std::vector<std::uint32_t>& depth_offsets,
                                      const std::vector<std::uint8_t>& codes_small_by_local,
                                      const std::vector<float>& coeffs_by_local,
                                      const std::vector<std::uint8_t>& virt_codes_full,
                                      const std::vector<float>& virt_coeffs_full,
                                      int n_real,
                                      int n_virt,
                                      std::string* err) {
    if (cid != next_cid_) {
        if (err) *err = "LinkageVirtualWriter::AppendCluster: cid out of order.";
        return false;
    }
    if (n_real < 0 || n_virt < 0) {
        if (err) *err = "LinkageVirtualWriter::AppendCluster: invalid sizes.";
        return false;
    }
    if (n_real == 0) {
        // Still advance offsets.
        real_offsets_out_.write(reinterpret_cast<const char*>(&real_count_), sizeof(real_count_));
        virt_offsets_out_.write(reinterpret_cast<const char*>(&virt_count_), sizeof(virt_count_));
        depth_offsets_offsets_out_.write(reinterpret_cast<const char*>(&depth_offsets_count_), sizeof(depth_offsets_count_));
        if (!real_offsets_out_ || !virt_offsets_out_ || !depth_offsets_offsets_out_) {
            if (err) *err = "LinkageVirtualWriter: offset write failed.";
            return false;
        }
        next_cid_ += 1;
        return true;
    }

    if (static_cast<int>(indices_depth_order.size()) != n_real ||
        static_cast<int>(parent_local_1based.size()) != n_real ||
        real_global_ids_by_local.size() != static_cast<std::size_t>(n_real)) {
        if (err) *err = "LinkageVirtualWriter::AppendCluster: real vectors size mismatch.";
        return false;
    }
    if (codes_small_by_local.size() != static_cast<std::size_t>(cfg_.m_codes) * static_cast<std::size_t>(n_real) ||
        coeffs_by_local.size() != static_cast<std::size_t>(cfg_.m) * static_cast<std::size_t>(n_real)) {
        if (err) *err = "LinkageVirtualWriter::AppendCluster: codes/coeffs size mismatch.";
        return false;
    }
    if (n_virt > 0) {
        if (virt_codes_full.size() != static_cast<std::size_t>(cfg_.m) * static_cast<std::size_t>(n_virt) ||
            virt_coeffs_full.size() != static_cast<std::size_t>(cfg_.m) * static_cast<std::size_t>(n_virt)) {
            if (err) *err = "LinkageVirtualWriter::AppendCluster: virt codes/coeffs size mismatch.";
            return false;
        }
    }

    // Write depth_offsets (variable length).
    const auto depth_len = static_cast<std::uint64_t>(depth_offsets.size());
    if (!WriteAll(&depth_offsets_out_, depth_offsets.data(), depth_offsets.size(), err)) {
        return false;
    }
    depth_offsets_count_ += depth_len;

    // Write real ids and parent in depth order.
    for (int pos = 0; pos < n_real; ++pos) {
        const int local = indices_depth_order[static_cast<std::size_t>(pos)];
        const std::uint32_t gid = real_global_ids_by_local[static_cast<std::size_t>(local)];
        real_ids_out_.write(reinterpret_cast<const char*>(&gid), sizeof(std::uint32_t));
        const std::uint32_t p = parent_local_1based[static_cast<std::size_t>(pos)];
        parent_out_.write(reinterpret_cast<const char*>(&p), sizeof(std::uint32_t));
    }
    if (!real_ids_out_ || !parent_out_) {
        if (err) *err = "LinkageVirtualWriter: real ids/parent write failed.";
        return false;
    }

    // codes_small: write columns in depth order (col-major).
    for (int pos = 0; pos < n_real; ++pos) {
        const int local = indices_depth_order[static_cast<std::size_t>(pos)];
        const std::size_t off = static_cast<std::size_t>(local) * static_cast<std::size_t>(cfg_.m_codes);
        codes_out_.write(reinterpret_cast<const char*>(codes_small_by_local.data() + off),
                         static_cast<std::streamsize>(cfg_.m_codes));
    }
    if (!codes_out_) {
        if (err) *err = "LinkageVirtualWriter: codes write failed.";
        return false;
    }

    // coeffs: write columns in depth order (col-major).
    for (int pos = 0; pos < n_real; ++pos) {
        const int local = indices_depth_order[static_cast<std::size_t>(pos)];
        const std::size_t off = static_cast<std::size_t>(local) * static_cast<std::size_t>(cfg_.m);
        coeffs_out_.write(reinterpret_cast<const char*>(coeffs_by_local.data() + off),
                          static_cast<std::streamsize>(cfg_.m * sizeof(float)));
    }
    if (!coeffs_out_) {
        if (err) *err = "LinkageVirtualWriter: coeffs write failed.";
        return false;
    }

    // Virtual nodes (if any) are stored in cluster-local order 0..n_virt-1.
    if (n_virt > 0) {
        virt_codes_out_.write(reinterpret_cast<const char*>(virt_codes_full.data()),
                              static_cast<std::streamsize>(virt_codes_full.size()));
        virt_coeffs_out_.write(reinterpret_cast<const char*>(virt_coeffs_full.data()),
                               static_cast<std::streamsize>(virt_coeffs_full.size() * sizeof(float)));
        if (!virt_codes_out_ || !virt_coeffs_out_) {
            if (err) *err = "LinkageVirtualWriter: virt write failed.";
            return false;
        }
    }

    real_count_ += static_cast<std::uint64_t>(n_real);
    virt_count_ += static_cast<std::uint64_t>(n_virt);

    // Append prefix sums.
    real_offsets_out_.write(reinterpret_cast<const char*>(&real_count_), sizeof(real_count_));
    virt_offsets_out_.write(reinterpret_cast<const char*>(&virt_count_), sizeof(virt_count_));
    depth_offsets_offsets_out_.write(reinterpret_cast<const char*>(&depth_offsets_count_), sizeof(depth_offsets_count_));
    if (!real_offsets_out_ || !virt_offsets_out_ || !depth_offsets_offsets_out_) {
        if (err) *err = "LinkageVirtualWriter: offsets write failed.";
        return false;
    }

    next_cid_ += 1;
    return true;
}

bool LinkageVirtualWriter::Finish(std::string* err) const {
    // Ensure all clusters were appended.
    if (next_cid_ != cfg_.nlist) {
        if (err) *err = "LinkageVirtualWriter::Finish: not all clusters were written.";
        return false;
    }
    return true;
}

}  // namespace stlq::io

