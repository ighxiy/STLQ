#include "stlq/io/linkage_list_store.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <system_error>

#include "stlq/common/logger.h"
#include "stlq/succinct/parent_louds.h"

#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace stlq::io {

namespace {

std::string JoinPath(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    if (a.back() == '/' || a.back() == '\\') return a + b;
    return a + "/" + b;
}

bool EnsureDir(const std::string& dir, std::string* err) {
    try {
        std::filesystem::create_directories(dir);
        return true;
    } catch (...) {
        if (err) *err = "LinkageListStore: failed to create dir: " + dir;
        return false;
    }
}

template <typename T>
bool WriteAll(std::ofstream* out, const T* ptr, std::size_t n, std::string* err) {
    if (n == 0) return true;
    out->write(reinterpret_cast<const char*>(ptr), static_cast<std::streamsize>(n * sizeof(T)));
    if (!(*out)) {
        if (err) *err = "LinkageListStore: write failed.";
        return false;
    }
    return true;
}

#if !defined(_WIN32)
bool PwriteAll(int fd, const void* buf, std::size_t bytes, std::uint64_t off, std::string* err) {
    const auto* p = static_cast<const std::uint8_t*>(buf);
    std::size_t remaining = bytes;
    std::uint64_t cur = off;
    while (remaining > 0) {
        const ssize_t w = ::pwrite(fd, p, remaining, static_cast<off_t>(cur));
        if (w < 0) {
            if (err) {
                *err = "LinkageListRandomWriter: pwrite failed: " + std::string(std::strerror(errno));
            }
            return false;
        }
        if (w == 0) {
            if (err) *err = "LinkageListRandomWriter: pwrite returned 0.";
            return false;
        }
        remaining -= static_cast<std::size_t>(w);
        p += w;
        cur += static_cast<std::uint64_t>(w);
    }
    return true;
}

int OpenRwTrunc(const std::string& path, std::string* err) {
    const int fd = ::open(path.c_str(), O_CREAT | O_RDWR, 0644);
    if (fd < 0) {
        if (err) {
            *err = "LinkageListRandomWriter: open failed: " + path + " : " + std::string(std::strerror(errno));
        }
        return -1;
    }
    return fd;
}

int OpenRwNoTrunc(const std::string& path, std::string* err) {
    const int fd = ::open(path.c_str(), O_RDWR, 0644);
    if (fd < 0) {
        if (err) {
            *err = "LinkageListRandomWriter: open(no-trunc) failed: " + path + " : " +
                   std::string(std::strerror(errno));
        }
        return -1;
    }
    return fd;
}

void CloseFd(int* fd) {
    if (fd && *fd >= 0) {
        ::close(*fd);
        *fd = -1;
    }
}

#endif  // !_WIN32

}  // namespace

bool LinkageListWriter::Open(const LinkageListStoreConfig& cfg, std::string* err) {
    cfg_ = cfg;
    real_count_ = 0;
    virt_count_ = 0;
    depth_offsets_count_ = 0;
    next_cid_ = 0;

    if (cfg_.dir.empty() || cfg_.nlist <= 0 || cfg_.m_codes < 0 ||
        (cfg_.code0_width_bytes != 1 && cfg_.code0_width_bytes != 2 && cfg_.code0_width_bytes != 4) ||
        (!cfg_.store_parent_u32 && !cfg_.store_parent_louds) ||
        (cfg_.store_parent_louds &&
         (cfg_.parent_louds_select_stride <= 0 ||
          cfg_.parent_louds_rank_words_per_super_log2 <= 0 ||
          cfg_.parent_louds_rank_words_per_super_log2 > 10))) {
        if (err) *err = "LinkageListWriter::Open: invalid config.";
        return false;
    }
    if (!EnsureDir(cfg_.dir, err)) {
        return false;
    }

    real_offsets_out_.open(JoinPath(cfg_.dir, "real_offsets.u64"), std::ios::binary | std::ios::trunc);
    virt_offsets_out_.open(JoinPath(cfg_.dir, "virt_offsets.u64"), std::ios::binary | std::ios::trunc);
    depth_offsets_offsets_out_.open(JoinPath(cfg_.dir, "depth_offsets_offsets.u64"),
                                    std::ios::binary | std::ios::trunc);
    depth_offsets_out_.open(JoinPath(cfg_.dir, "depth_offsets.u32"), std::ios::binary | std::ios::trunc);
    real_ids_out_.open(JoinPath(cfg_.dir, "real_ids.u32"), std::ios::binary | std::ios::trunc);
    if (cfg_.store_parent_u32) {
        parent_out_.open(JoinPath(cfg_.dir, "parent.u32"), std::ios::binary | std::ios::trunc);
    }
    if (cfg_.store_parent_louds) {
        parent_louds_offsets_out_.open(JoinPath(cfg_.dir, "parent_louds_offsets.u64"),
                                       std::ios::binary | std::ios::trunc);
        parent_louds_out_.open(JoinPath(cfg_.dir, "parent_louds.bin"),
                               std::ios::binary | std::ios::trunc);
    }
    codes_out_.open(JoinPath(cfg_.dir, "codes.bin"), std::ios::binary | std::ios::trunc);
    code0_one_out_.open(JoinPath(cfg_.dir, "code0_one.bin"), std::ios::binary | std::ios::trunc);
    virt_codes_out_.open(JoinPath(cfg_.dir, "virt_codes.bin"), std::ios::binary | std::ios::trunc);
    if (cfg_.store_coeffs_f32) {
        coeffs_out_.open(JoinPath(cfg_.dir, "coeffs.f32"), std::ios::binary | std::ios::trunc);
        a0_out_.open(JoinPath(cfg_.dir, "a0.f32"), std::ios::binary | std::ios::trunc);
        virt_coeffs_out_.open(JoinPath(cfg_.dir, "virt_coeffs.f32"), std::ios::binary | std::ios::trunc);
        virt_a0_out_.open(JoinPath(cfg_.dir, "virt_a0.f32"), std::ios::binary | std::ios::trunc);
    }

    if (!real_offsets_out_.is_open() || !virt_offsets_out_.is_open() || !depth_offsets_offsets_out_.is_open() ||
        !depth_offsets_out_.is_open() || !real_ids_out_.is_open() ||
        (cfg_.store_parent_u32 && !parent_out_.is_open()) ||
        (cfg_.store_parent_louds && (!parent_louds_offsets_out_.is_open() || !parent_louds_out_.is_open())) ||
        !codes_out_.is_open() || !code0_one_out_.is_open() || !virt_codes_out_.is_open() ||
        (cfg_.store_coeffs_f32 && (!coeffs_out_.is_open() || !a0_out_.is_open() ||
                                   !virt_coeffs_out_.is_open() || !virt_a0_out_.is_open()))) {
        if (err) *err = "LinkageListWriter::Open: failed to open output files.";
        return false;
    }

    const std::uint64_t z = 0;
    real_offsets_out_.write(reinterpret_cast<const char*>(&z), sizeof(z));
    virt_offsets_out_.write(reinterpret_cast<const char*>(&z), sizeof(z));
    depth_offsets_offsets_out_.write(reinterpret_cast<const char*>(&z), sizeof(z));
    if (cfg_.store_parent_louds) {
        parent_louds_offsets_out_.write(reinterpret_cast<const char*>(&z), sizeof(z));
        parent_louds_bytes_ = 0;
    }
    if (!real_offsets_out_ || !virt_offsets_out_ || !depth_offsets_offsets_out_ ||
        (cfg_.store_parent_louds && !parent_louds_offsets_out_)) {
        if (err) *err = "LinkageListWriter::Open: failed to write initial offsets.";
        return false;
    }
    return true;
}

bool LinkageListWriter::AppendCluster(int cid,
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
                                    int n_real,
                                    int n_virt,
                                    std::string* err) {
    if (cid != next_cid_) {
        if (err) *err = "LinkageListWriter::AppendCluster: cid out of order.";
        return false;
    }
    if (n_real < 0 || n_virt < 0) {
        if (err) *err = "LinkageListWriter::AppendCluster: invalid n_real/n_virt.";
        return false;
    }

    if (!depth_offsets.empty()) {
        if (depth_offsets.back() != static_cast<std::uint32_t>(n_real)) {
            if (err) *err = "LinkageListWriter::AppendCluster: depth_offsets.back != n_real.";
            return false;
        }
    }

    const std::size_t codes_bytes_expect =
        static_cast<std::size_t>(cfg_.m_codes) * static_cast<std::size_t>(n_real) *
        static_cast<std::size_t>(LinkageListStoreConfig::kSmallCodeWidthBytes);
    const std::size_t coeffs_expect =
        cfg_.store_coeffs_f32 ? static_cast<std::size_t>(cfg_.m_codes) * static_cast<std::size_t>(n_real) : 0;
    const std::size_t virt_codes_bytes_expect =
        static_cast<std::size_t>(cfg_.m_codes) * static_cast<std::size_t>(n_virt) *
        static_cast<std::size_t>(LinkageListStoreConfig::kSmallCodeWidthBytes);
    const std::size_t virt_coeffs_expect =
        cfg_.store_coeffs_f32 ? static_cast<std::size_t>(cfg_.m_codes) * static_cast<std::size_t>(n_virt) : 0;
    const std::size_t code0_one_bytes_expect =
        static_cast<std::size_t>(n_real) * static_cast<std::size_t>(cfg_.code0_width_bytes);

    if (real_ids_depth_order.size() != static_cast<std::size_t>(n_real) ||
        parent_local_1based_depth_order.size() != static_cast<std::size_t>(n_real) ||
        codes_small_depth_order_bytes.size() != codes_bytes_expect ||
        (cfg_.store_coeffs_f32 && coeffs_small_depth_order.size() != coeffs_expect) ||
        code0_one_depth_order_bytes.size() != code0_one_bytes_expect ||
        (cfg_.store_coeffs_f32 && a0_depth_order.size() != static_cast<std::size_t>(n_real)) ||
        virt_codes_small_bytes.size() != virt_codes_bytes_expect ||
        (cfg_.store_coeffs_f32 && virt_coeffs_small.size() != virt_coeffs_expect) ||
        (cfg_.store_coeffs_f32 && virt_a0.size() != static_cast<std::size_t>(n_virt))) {
        if (err) *err = "LinkageListWriter::AppendCluster: payload size mismatch.";
        return false;
    }

    if (!WriteAll(&depth_offsets_out_, depth_offsets.data(), depth_offsets.size(), err)) {
        return false;
    }
    depth_offsets_count_ += static_cast<std::uint64_t>(depth_offsets.size());

    if (!WriteAll(&real_ids_out_, real_ids_depth_order.data(), real_ids_depth_order.size(), err)) {
        return false;
    }
    if (cfg_.store_parent_u32) {
        if (!WriteAll(&parent_out_, parent_local_1based_depth_order.data(),
                      parent_local_1based_depth_order.size(), err)) {
            return false;
        }
    }

    if (cfg_.store_parent_louds) {
        const std::size_t n_total = static_cast<std::size_t>(std::max(0, n_real) + std::max(0, n_virt));
        std::vector<std::uint32_t> parent_all(n_total, 0u);
        const std::size_t n_virt_sz = static_cast<std::size_t>(std::max(0, n_virt));
        const std::size_t n_real_sz = static_cast<std::size_t>(std::max(0, n_real));
        std::memcpy(parent_all.data() + n_virt_sz,
                    parent_local_1based_depth_order.data(),
                    n_real_sz * sizeof(std::uint32_t));
        stlq::succinct::ParentLOUDS louds;
        louds.BuildFromParent1Based(parent_all.data(),
                                    parent_all.size(),
                                    static_cast<std::uint32_t>(std::max(1, cfg_.parent_louds_select_stride)),
                                    static_cast<std::uint32_t>(std::max(1, cfg_.parent_louds_rank_words_per_super_log2)),
                                    /*build_indices=*/false);
        const std::vector<std::uint8_t> blob = louds.Serialize();
        parent_louds_out_.write(reinterpret_cast<const char*>(blob.data()),
                                static_cast<std::streamsize>(blob.size()));
        if (!parent_louds_out_) {
            if (err) *err = "LinkageListWriter::AppendCluster: parent_louds write failed.";
            return false;
        }
        parent_louds_bytes_ += static_cast<std::uint64_t>(blob.size());
        parent_louds_offsets_out_.write(reinterpret_cast<const char*>(&parent_louds_bytes_),
                                        sizeof(parent_louds_bytes_));
        if (!parent_louds_offsets_out_) {
            if (err) *err = "LinkageListWriter::AppendCluster: parent_louds_offsets write failed.";
            return false;
        }
    }

    if (!codes_small_depth_order_bytes.empty()) {
        codes_out_.write(reinterpret_cast<const char*>(codes_small_depth_order_bytes.data()),
                         static_cast<std::streamsize>(codes_small_depth_order_bytes.size()));
        if (!codes_out_) {
            if (err) *err = "LinkageListWriter::AppendCluster: codes write failed.";
            return false;
        }
    }
    if (cfg_.store_coeffs_f32) {
        if (!coeffs_small_depth_order.empty()) {
            coeffs_out_.write(reinterpret_cast<const char*>(coeffs_small_depth_order.data()),
                              static_cast<std::streamsize>(coeffs_small_depth_order.size() * sizeof(float)));
            if (!coeffs_out_) {
                if (err) *err = "LinkageListWriter::AppendCluster: coeffs write failed.";
                return false;
            }
        }
    }
    if (!code0_one_depth_order_bytes.empty()) {
        code0_one_out_.write(reinterpret_cast<const char*>(code0_one_depth_order_bytes.data()),
                             static_cast<std::streamsize>(code0_one_depth_order_bytes.size()));
        if (!code0_one_out_) {
            if (err) *err = "LinkageListWriter::AppendCluster: code0_one write failed.";
            return false;
        }
    }
    if (cfg_.store_coeffs_f32) {
        if (!a0_depth_order.empty()) {
            a0_out_.write(reinterpret_cast<const char*>(a0_depth_order.data()),
                          static_cast<std::streamsize>(a0_depth_order.size() * sizeof(float)));
            if (!a0_out_) {
                if (err) *err = "LinkageListWriter::AppendCluster: a0 write failed.";
                return false;
            }
        }
    }

    if (!virt_codes_small_bytes.empty()) {
        virt_codes_out_.write(reinterpret_cast<const char*>(virt_codes_small_bytes.data()),
                              static_cast<std::streamsize>(virt_codes_small_bytes.size()));
        if (!virt_codes_out_) {
            if (err) *err = "LinkageListWriter::AppendCluster: virt codes write failed.";
            return false;
        }
    }
    if (cfg_.store_coeffs_f32) {
        if (!virt_coeffs_small.empty()) {
            virt_coeffs_out_.write(reinterpret_cast<const char*>(virt_coeffs_small.data()),
                                   static_cast<std::streamsize>(virt_coeffs_small.size() * sizeof(float)));
            if (!virt_coeffs_out_) {
                if (err) *err = "LinkageListWriter::AppendCluster: virt coeffs write failed.";
                return false;
            }
        }
        if (!virt_a0.empty()) {
            virt_a0_out_.write(reinterpret_cast<const char*>(virt_a0.data()),
                               static_cast<std::streamsize>(virt_a0.size() * sizeof(float)));
            if (!virt_a0_out_) {
                if (err) *err = "LinkageListWriter::AppendCluster: virt a0 write failed.";
                return false;
            }
        }
    }

    real_count_ += static_cast<std::uint64_t>(n_real);
    virt_count_ += static_cast<std::uint64_t>(n_virt);

    real_offsets_out_.write(reinterpret_cast<const char*>(&real_count_), sizeof(real_count_));
    virt_offsets_out_.write(reinterpret_cast<const char*>(&virt_count_), sizeof(virt_count_));
    depth_offsets_offsets_out_.write(reinterpret_cast<const char*>(&depth_offsets_count_),
                                     sizeof(depth_offsets_count_));
    if (!real_offsets_out_ || !virt_offsets_out_ || !depth_offsets_offsets_out_) {
        if (err) *err = "LinkageListWriter::AppendCluster: offsets write failed.";
        return false;
    }

    next_cid_ += 1;
    return true;
}

bool LinkageListWriter::WriteMeta(std::string* err) const {
    struct Header {
        char magic[8]{};
        std::uint32_t version = 4;
        std::uint32_t nlist = 0;
        std::uint32_t m_codes = 0;
        std::uint32_t small_code_width_bytes = 0;
        std::uint32_t code0_width_bytes = 0;
        std::uint32_t reserved[8] = {0};
    } h;
    std::memcpy(h.magic, "LINKAGELST", 8);
    h.nlist = static_cast<std::uint32_t>(cfg_.nlist);
    h.m_codes = static_cast<std::uint32_t>(cfg_.m_codes);
    h.small_code_width_bytes = static_cast<std::uint32_t>(LinkageListStoreConfig::kSmallCodeWidthBytes);
    h.code0_width_bytes = static_cast<std::uint32_t>(cfg_.code0_width_bytes);
    h.reserved[0] = cfg_.store_coeffs_f32 ? 1u : 0u;
    h.reserved[4] = cfg_.store_parent_u32 ? 1u : 0u;
    h.reserved[1] = cfg_.store_parent_louds ? 1u : 0u;
    h.reserved[2] = static_cast<std::uint32_t>(std::max(1, cfg_.parent_louds_select_stride));
    h.reserved[3] = static_cast<std::uint32_t>(std::max(1, cfg_.parent_louds_rank_words_per_super_log2));

    const std::string path = JoinPath(cfg_.dir, "meta.bin");
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        if (err) *err = "LinkageListWriter::Finish: failed to open meta.bin.";
        return false;
    }
    out.write(reinterpret_cast<const char*>(&h), sizeof(Header));
    if (!out) {
        if (err) *err = "LinkageListWriter::Finish: failed to write meta.bin.";
        return false;
    }
    return true;
}

bool LinkageListWriter::Finish(std::string* err) {
    if (next_cid_ != cfg_.nlist) {
        if (err) *err = "LinkageListWriter::Finish: not all clusters were written.";
        return false;
    }
    // Ensure data is visible to readers in the same process (we may open the store right after writing).
    real_offsets_out_.flush();
    virt_offsets_out_.flush();
    depth_offsets_offsets_out_.flush();
    depth_offsets_out_.flush();
    real_ids_out_.flush();
    if (cfg_.store_parent_u32) {
        parent_out_.flush();
    }
    if (cfg_.store_parent_louds) {
        parent_louds_offsets_out_.flush();
        parent_louds_out_.flush();
    }
    codes_out_.flush();
    code0_one_out_.flush();
    virt_codes_out_.flush();
    if (cfg_.store_coeffs_f32) {
        coeffs_out_.flush();
        a0_out_.flush();
        virt_coeffs_out_.flush();
        virt_a0_out_.flush();
    }

    if (!real_offsets_out_ || !virt_offsets_out_ || !depth_offsets_offsets_out_ || !depth_offsets_out_ ||
        !real_ids_out_ || (cfg_.store_parent_u32 && !parent_out_) ||
        (cfg_.store_parent_louds && (!parent_louds_offsets_out_ || !parent_louds_out_)) ||
        !codes_out_ || !code0_one_out_ || !virt_codes_out_ ||
        (cfg_.store_coeffs_f32 && (!coeffs_out_ || !a0_out_ || !virt_coeffs_out_ || !virt_a0_out_))) {
        if (err) *err = "LinkageListWriter::Finish: flush failed.";
        return false;
    }

    real_offsets_out_.close();
    virt_offsets_out_.close();
    depth_offsets_offsets_out_.close();
    depth_offsets_out_.close();
    real_ids_out_.close();
    if (cfg_.store_parent_u32) {
        parent_out_.close();
    }
    if (cfg_.store_parent_louds) {
        parent_louds_offsets_out_.close();
        parent_louds_out_.close();
    }
    codes_out_.close();
    code0_one_out_.close();
    virt_codes_out_.close();
    if (cfg_.store_coeffs_f32) {
        coeffs_out_.close();
        a0_out_.close();
        virt_coeffs_out_.close();
        virt_a0_out_.close();
    }

    if (!real_offsets_out_ || !virt_offsets_out_ || !depth_offsets_offsets_out_ || !depth_offsets_out_ ||
        !real_ids_out_ || (cfg_.store_parent_u32 && !parent_out_) ||
        (cfg_.store_parent_louds && (!parent_louds_offsets_out_ || !parent_louds_out_)) ||
        !codes_out_ || !code0_one_out_ || !virt_codes_out_ ||
        (cfg_.store_coeffs_f32 && (!coeffs_out_ || !a0_out_ || !virt_coeffs_out_ || !virt_a0_out_))) {
        if (err) *err = "LinkageListWriter::Finish: close failed.";
        return false;
    }
    return WriteMeta(err);
}

bool LinkageListReader::ReadAll(std::ifstream& in, void* dst, std::size_t bytes) {
    in.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes));
    return static_cast<std::size_t>(in.gcount()) == bytes;
}

bool LinkageListReader::Open(const std::string& dir, std::string* err) {
    dir_ = dir;
    meta_ = {};
    real_offsets_.clear();
    virt_offsets_.clear();
    depth_offsets_offsets_.clear();
    parent_louds_offsets_bytes_.clear();

    if (dir_.empty()) {
        if (err) *err = "LinkageListReader::Open: empty dir.";
        return false;
    }

    const std::string meta_path = JoinPath(dir_, "meta.bin");
    std::ifstream meta_in(meta_path, std::ios::binary);
    if (!meta_in.is_open()) {
        if (err) *err = "LinkageListReader::Open: failed to open " + meta_path;
        return false;
    }
    struct HeaderPrefix {
        char magic[8]{};
        std::uint32_t version = 0;
        std::uint32_t nlist = 0;
        std::uint32_t m_codes = 0;
        std::uint32_t small_code_width_bytes = 0;
    } pfx;
    if (!ReadAll(meta_in, &pfx, sizeof(HeaderPrefix))) {
        if (err) *err = "LinkageListReader::Open: failed to read meta.bin header.";
        return false;
    }
    const bool is_linkage_magic = std::memcmp(pfx.magic, "LINKAGELST", 8) == 0;
    // CHAINLST is the pre-rename spelling of the same on-disk layout.  Keep the
    // writer canonical, but accept archived stores produced before the rename.
    const bool is_legacy_chain_magic = std::memcmp(pfx.magic, "CHAINLST", 8) == 0;
    if ((!is_linkage_magic && !is_legacy_chain_magic) ||
        (pfx.version != 2 && pfx.version != 3 && pfx.version != 4)) {
        if (err) *err = "LinkageListReader::Open: unsupported meta.bin.";
        return false;
    }
    if (pfx.nlist == 0 || pfx.small_code_width_bytes != LinkageListStoreConfig::kSmallCodeWidthBytes) {
        if (err) *err = "LinkageListReader::Open: invalid meta fields.";
        return false;
    }
    if (pfx.version == 4) {
        std::uint32_t code0_width_bytes = 0;
        std::uint32_t reserved[8] = {0};
        if (!ReadAll(meta_in, &code0_width_bytes, sizeof(code0_width_bytes)) ||
            !ReadAll(meta_in, reserved, sizeof(reserved))) {
            if (err) *err = "LinkageListReader::Open: failed to read meta.bin header.";
            return false;
        }
        if (code0_width_bytes != 1 && code0_width_bytes != 2 && code0_width_bytes != 4) {
            if (err) *err = "LinkageListReader::Open: invalid meta fields.";
            return false;
        }
        meta_.code0_width_bytes = static_cast<int>(code0_width_bytes);
        meta_.store_coeffs_f32 = (reserved[0] != 0) ? 1 : 0;
        meta_.store_parent_louds = (reserved[1] != 0) ? 1 : 0;
        meta_.parent_louds_select_stride = reserved[2] ? static_cast<int>(reserved[2]) : 128;
        meta_.parent_louds_rank_words_per_super_log2 = reserved[3] ? static_cast<int>(reserved[3]) : 4;
        meta_.store_parent_u32 = (reserved[4] != 0) ? 1 : 0;
    } else if (pfx.version == 3) {
        std::uint32_t code0_width_bytes = 0;
        std::uint32_t reserved[8] = {0};
        if (!ReadAll(meta_in, &code0_width_bytes, sizeof(code0_width_bytes)) ||
            !ReadAll(meta_in, reserved, sizeof(reserved))) {
            if (err) *err = "LinkageListReader::Open: failed to read meta.bin header.";
            return false;
        }
        if (code0_width_bytes != 1 && code0_width_bytes != 2 && code0_width_bytes != 4) {
            if (err) *err = "LinkageListReader::Open: invalid meta fields.";
            return false;
        }
        meta_.code0_width_bytes = static_cast<int>(code0_width_bytes);
        meta_.store_coeffs_f32 = 1;
        meta_.store_parent_louds = (reserved[1] != 0) ? 1 : 0;
        meta_.parent_louds_select_stride = reserved[2] ? static_cast<int>(reserved[2]) : 128;
        meta_.parent_louds_rank_words_per_super_log2 = reserved[3] ? static_cast<int>(reserved[3]) : 4;
        meta_.store_parent_u32 = 1;
    } else {
        // v2: code0_one uses the same width as codes_small.
        std::uint32_t reserved[8] = {0};
        if (!ReadAll(meta_in, reserved, sizeof(reserved))) {
            if (err) *err = "LinkageListReader::Open: failed to read meta.bin header.";
            return false;
        }
        meta_.code0_width_bytes = static_cast<int>(pfx.small_code_width_bytes);
        meta_.store_coeffs_f32 = 1;
        meta_.store_parent_louds = (reserved[1] != 0) ? 1 : 0;
        meta_.parent_louds_select_stride = reserved[2] ? static_cast<int>(reserved[2]) : 128;
        meta_.parent_louds_rank_words_per_super_log2 = reserved[3] ? static_cast<int>(reserved[3]) : 4;
        meta_.store_parent_u32 = 1;
    }
    meta_.nlist = static_cast<int>(pfx.nlist);
    meta_.m_codes = static_cast<int>(pfx.m_codes);
    meta_.small_code_width_bytes = static_cast<int>(pfx.small_code_width_bytes);

    auto load_offsets = [&](const std::string& name, std::vector<std::uint64_t>* outv) -> bool {
        const std::string path = JoinPath(dir_, name);
        std::ifstream in(path, std::ios::binary);
        if (!in.is_open()) {
            if (err) *err = "LinkageListReader::Open: failed to open " + path;
            return false;
        }
        outv->resize(static_cast<std::size_t>(meta_.nlist) + 1);
        in.read(reinterpret_cast<char*>(outv->data()),
                static_cast<std::streamsize>(outv->size() * sizeof(std::uint64_t)));
        if (!in) {
            if (err) *err = "LinkageListReader::Open: failed to read " + path;
            return false;
        }
        return true;
    };

    auto load_offsets_quiet = [&](const std::string& name, std::vector<std::uint64_t>* outv) -> bool {
        const std::string path = JoinPath(dir_, name);
        std::ifstream in(path, std::ios::binary);
        if (!in.is_open()) {
            return false;
        }
        outv->resize(static_cast<std::size_t>(meta_.nlist) + 1);
        in.read(reinterpret_cast<char*>(outv->data()),
                static_cast<std::streamsize>(outv->size() * sizeof(std::uint64_t)));
        return static_cast<bool>(in);
    };

    if (!load_offsets("real_offsets.u64", &real_offsets_)) return false;
    if (!load_offsets("virt_offsets.u64", &virt_offsets_)) return false;
    if (!load_offsets("depth_offsets_offsets.u64", &depth_offsets_offsets_)) return false;

    // LOUDS is an optional add-on. Prefer meta flag, but be backward compatible:
    // some early stores wrote `parent_louds.bin` / `parent_louds_offsets.u64` but forgot to set meta.reserved[1].
    if (meta_.store_parent_louds) {
        std::vector<std::uint64_t> tmp;
        if (!load_offsets_quiet("parent_louds_offsets.u64", &tmp)) {
            meta_.store_parent_louds = 0;
            parent_louds_offsets_bytes_.clear();
        } else {
            parent_louds_offsets_bytes_ = std::move(tmp);
        }
    } else {
        const std::string blob_path = JoinPath(dir_, "parent_louds.bin");
        std::ifstream blob_in(blob_path, std::ios::binary);
        if (blob_in.is_open()) {
            std::vector<std::uint64_t> tmp;
            if (load_offsets_quiet("parent_louds_offsets.u64", &tmp)) {
                meta_.store_parent_louds = 1;
                meta_.parent_louds_select_stride = 128;
                meta_.parent_louds_rank_words_per_super_log2 = 4;
                parent_louds_offsets_bytes_ = std::move(tmp);
            }
        }
    }

    return true;
}

bool LinkageListReader::ReadClusterSpan(int cid,
                                      std::uint64_t* real_lo,
                                      std::uint64_t* real_hi,
                                      std::uint64_t* virt_lo,
                                      std::uint64_t* virt_hi,
                                      std::uint64_t* depth_lo,
                                      std::uint64_t* depth_hi,
                                      std::string* err) const {
    if (cid < 0 || cid >= meta_.nlist) {
        if (err) *err = "LinkageListReader::ReadClusterSpan: cid out of range.";
        return false;
    }
    const auto i = static_cast<std::size_t>(cid);
    if (real_lo) *real_lo = real_offsets_[i];
    if (real_hi) *real_hi = real_offsets_[i + 1];
    if (virt_lo) *virt_lo = virt_offsets_[i];
    if (virt_hi) *virt_hi = virt_offsets_[i + 1];
    if (depth_lo) *depth_lo = depth_offsets_offsets_[i];
    if (depth_hi) *depth_hi = depth_offsets_offsets_[i + 1];
    return true;
}

bool LinkageListReader::ReadU32Span(const std::string& path,
                                 std::uint64_t begin,
                                 std::uint32_t count,
                                 std::vector<std::uint32_t>* out,
                                 std::string* err) {
    out->clear();
    if (count == 0) return true;
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        if (err) *err = "LinkageListReader: failed to open " + path;
        return false;
    }
    const std::uint64_t off = begin * sizeof(std::uint32_t);
    in.seekg(static_cast<std::streamoff>(off), std::ios::beg);
    if (!in) {
        if (err) *err = "LinkageListReader: seek failed.";
        return false;
    }
    out->resize(count);
    in.read(reinterpret_cast<char*>(out->data()), static_cast<std::streamsize>(count * sizeof(std::uint32_t)));
    if (!in) {
        if (err) *err = "LinkageListReader: read failed.";
        return false;
    }
    return true;
}

bool LinkageListReader::ReadF32Span(const std::string& path,
                                 std::uint64_t begin,
                                 std::uint32_t count,
                                 std::vector<float>* out,
                                 std::string* err) {
    out->clear();
    if (count == 0) return true;
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        if (err) *err = "LinkageListReader: failed to open " + path;
        return false;
    }
    const std::uint64_t off = begin * sizeof(float);
    in.seekg(static_cast<std::streamoff>(off), std::ios::beg);
    if (!in) {
        if (err) *err = "LinkageListReader: seek failed.";
        return false;
    }
    out->resize(count);
    in.read(reinterpret_cast<char*>(out->data()), static_cast<std::streamsize>(count * sizeof(float)));
    if (!in) {
        if (err) *err = "LinkageListReader: read failed.";
        return false;
    }
    return true;
}

bool LinkageListReader::ReadBytesSpan(const std::string& path,
                                   std::uint64_t begin_bytes,
                                   std::uint64_t bytes,
                                   std::vector<std::uint8_t>* out,
                                   std::string* err) {
    out->clear();
    if (bytes == 0) return true;
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        if (err) *err = "LinkageListReader: failed to open " + path;
        return false;
    }
    in.seekg(static_cast<std::streamoff>(begin_bytes), std::ios::beg);
    if (!in) {
        if (err) *err = "LinkageListReader: seek failed.";
        return false;
    }
    out->resize(static_cast<std::size_t>(bytes));
    in.read(reinterpret_cast<char*>(out->data()), static_cast<std::streamsize>(bytes));
    if (!in) {
        if (err) *err = "LinkageListReader: read failed.";
        return false;
    }
    return true;
}

bool LinkageListReader::ReadCluster(int cid,
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
                                  std::string* err) const {
    if (!real_ids_depth_order || !parent_local_1based_depth_order || !depth_offsets ||
        !codes_small_depth_order_bytes || !code0_one_depth_order_bytes || !virt_codes_small_bytes) {
        if (err) *err = "LinkageListReader::ReadCluster: required output is null.";
        return false;
    }
    std::uint64_t real_lo = 0, real_hi = 0, virt_lo = 0, virt_hi = 0, depth_lo = 0, depth_hi = 0;
    if (!ReadClusterSpan(cid, &real_lo, &real_hi, &virt_lo, &virt_hi, &depth_lo, &depth_hi, err)) {
        return false;
    }
    const std::uint64_t n_real64 = real_hi - real_lo;
    const std::uint64_t n_virt64 = virt_hi - virt_lo;
    if (n_real64 > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) ||
        n_virt64 > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
        if (err) *err = "LinkageListReader::ReadCluster: cluster too large.";
        return false;
    }
    const auto n_real = static_cast<std::uint32_t>(n_real64);
    const auto n_virt = static_cast<std::uint32_t>(n_virt64);
    const auto depth_len = static_cast<std::uint32_t>(depth_hi - depth_lo);

    const std::string ids_path = JoinPath(dir_, "real_ids.u32");
    const std::string parent_path = JoinPath(dir_, "parent.u32");
    const std::string depth_path = JoinPath(dir_, "depth_offsets.u32");
    const std::string codes_path = JoinPath(dir_, "codes.bin");
    const std::string code0_path = JoinPath(dir_, "code0_one.bin");
    const std::string vcodes_path = JoinPath(dir_, "virt_codes.bin");
    const std::string coeffs_path = JoinPath(dir_, "coeffs.f32");
    const std::string a0_path = JoinPath(dir_, "a0.f32");
    const std::string vcoeffs_path = JoinPath(dir_, "virt_coeffs.f32");
    const std::string va0_path = JoinPath(dir_, "virt_a0.f32");

    if (!ReadU32Span(ids_path, real_lo, n_real, real_ids_depth_order, err)) return false;
    if (parent_local_1based_depth_order) {
        if (meta_.store_parent_u32 != 0) {
            if (!ReadU32Span(parent_path, real_lo, n_real, parent_local_1based_depth_order, err)) return false;
        } else if (meta_.store_parent_louds != 0) {
            std::vector<std::uint8_t> blob;
            if (!ReadClusterParentLOUDSBlob(cid, &blob, err) || blob.empty()) {
                if (err && err->empty()) *err = "LinkageListReader::ReadCluster: missing parent.u32 and failed to read parent LOUDS.";
                return false;
            }
            try {
                stlq::succinct::ParentLOUDS louds;
                louds.Deserialize(blob.data(),
                                  blob.size(),
                                  static_cast<std::uint32_t>(std::max(1, meta_.parent_louds_rank_words_per_super_log2)),
                                  /*build_indices=*/false);
                std::vector<std::uint32_t> parent_all;
                const std::size_t nc = static_cast<std::size_t>(n_real) + static_cast<std::size_t>(n_virt);
                louds.DecodeParent1Based(&parent_all, nc);
                parent_local_1based_depth_order->resize(n_real);
                for (std::size_t i = 0; i < static_cast<std::size_t>(n_real); ++i) {
                    (*parent_local_1based_depth_order)[i] = parent_all[static_cast<std::size_t>(n_virt) + i];
                }
                static std::atomic<bool> warned{false};
                if (!warned.exchange(true)) {
                    LogWarn("linkage_list: parent.u32 not stored; decoding parent from parent_louds.bin on-demand.");
                }
            } catch (...) {
                if (err && err->empty()) *err = "LinkageListReader::ReadCluster: parent LOUDS decode failed.";
                return false;
            }
        } else {
            if (err) *err = "LinkageListReader::ReadCluster: parent requested but neither parent.u32 nor parent LOUDS is available.";
            return false;
        }
    }
    if (!ReadU32Span(depth_path, depth_lo, depth_len, depth_offsets, err)) return false;

    const std::uint64_t codes_bytes =
        static_cast<std::uint64_t>(meta_.m_codes) * static_cast<std::uint64_t>(n_real) *
        static_cast<std::uint64_t>(meta_.small_code_width_bytes);
    const std::uint64_t codes_off = real_lo * static_cast<std::uint64_t>(meta_.m_codes) *
                                   static_cast<std::uint64_t>(meta_.small_code_width_bytes);
    if (!ReadBytesSpan(codes_path, codes_off, codes_bytes, codes_small_depth_order_bytes, err)) return false;

    if (meta_.store_coeffs_f32 != 0) {
        if (coeffs_small_depth_order) {
            const std::uint64_t coeffs_count =
                static_cast<std::uint64_t>(meta_.m_codes) * static_cast<std::uint64_t>(n_real);
            const std::uint64_t coeffs_off = real_lo * static_cast<std::uint64_t>(meta_.m_codes);
            if (!ReadF32Span(coeffs_path, coeffs_off, static_cast<std::uint32_t>(coeffs_count),
                             coeffs_small_depth_order, err)) return false;
        }
    } else {
        if (coeffs_small_depth_order) coeffs_small_depth_order->clear();
    }

    const std::uint64_t code0_bytes = static_cast<std::uint64_t>(n_real) *
                                     static_cast<std::uint64_t>(meta_.code0_width_bytes);
    const std::uint64_t code0_off = real_lo * static_cast<std::uint64_t>(meta_.code0_width_bytes);
    if (!ReadBytesSpan(code0_path, code0_off, code0_bytes, code0_one_depth_order_bytes, err)) return false;

    if (meta_.store_coeffs_f32 != 0) {
        if (a0_depth_order) {
            if (!ReadF32Span(a0_path, real_lo, n_real, a0_depth_order, err)) return false;
        }
    } else {
        if (a0_depth_order) a0_depth_order->clear();
    }

    const std::uint64_t vcodes_bytes =
        static_cast<std::uint64_t>(meta_.m_codes) * static_cast<std::uint64_t>(n_virt) *
        static_cast<std::uint64_t>(meta_.small_code_width_bytes);
    const std::uint64_t vcodes_off = virt_lo * static_cast<std::uint64_t>(meta_.m_codes) *
                                    static_cast<std::uint64_t>(meta_.small_code_width_bytes);
    if (!ReadBytesSpan(vcodes_path, vcodes_off, vcodes_bytes, virt_codes_small_bytes, err)) return false;

    if (meta_.store_coeffs_f32 != 0) {
        if (virt_coeffs_small) {
            const std::uint64_t vcoeffs_count =
                static_cast<std::uint64_t>(meta_.m_codes) * static_cast<std::uint64_t>(n_virt);
            const std::uint64_t vcoeffs_off = virt_lo * static_cast<std::uint64_t>(meta_.m_codes);
            if (!ReadF32Span(vcoeffs_path, vcoeffs_off, static_cast<std::uint32_t>(vcoeffs_count),
                             virt_coeffs_small, err)) return false;
        }
        if (virt_a0) {
            if (!ReadF32Span(va0_path, virt_lo, n_virt, virt_a0, err)) return false;
        }
    } else {
        if (virt_coeffs_small) virt_coeffs_small->clear();
        if (virt_a0) virt_a0->clear();
    }

    return true;
}

bool LinkageListReader::ReadClusterDepthOffsets(int cid,
                                             std::vector<std::uint32_t>* depth_offsets,
                                             std::uint32_t* n_real_out,
                                             std::string* err) const {
    if (!depth_offsets) {
        if (err) *err = "LinkageListReader::ReadClusterDepthOffsets: depth_offsets output is null.";
        return false;
    }
    std::uint64_t real_lo = 0, real_hi = 0, virt_lo = 0, virt_hi = 0, depth_lo = 0, depth_hi = 0;
    if (!ReadClusterSpan(cid, &real_lo, &real_hi, &virt_lo, &virt_hi, &depth_lo, &depth_hi, err)) {
        return false;
    }
    const std::uint64_t n_real64 = real_hi - real_lo;
    if (n_real64 > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
        if (err) *err = "LinkageListReader::ReadClusterDepthOffsets: cluster too large.";
        return false;
    }
    const auto depth_len = static_cast<std::uint32_t>(depth_hi - depth_lo);
    const std::string depth_path = JoinPath(dir_, "depth_offsets.u32");
    if (!ReadU32Span(depth_path, depth_lo, depth_len, depth_offsets, err)) {
        return false;
    }
    if (n_real_out) {
        *n_real_out = static_cast<std::uint32_t>(n_real64);
    }
    return true;
}

bool LinkageListReader::ReadClusterParentLOUDSBlob(int cid,
                                                 std::vector<std::uint8_t>* blob,
                                                 std::string* err) const {
    if (!blob) {
        if (err) *err = "LinkageListReader::ReadClusterParentLOUDSBlob: blob output is null.";
        return false;
    }
    blob->clear();
    if (meta_.store_parent_louds == 0) {
        return true;
    }
    if (cid < 0 || cid >= meta_.nlist) {
        if (err) *err = "LinkageListReader::ReadClusterParentLOUDSBlob: cid out of range.";
        return false;
    }
    if (parent_louds_offsets_bytes_.size() != static_cast<std::size_t>(meta_.nlist) + 1) {
        if (err) *err = "LinkageListReader::ReadClusterParentLOUDSBlob: parent_louds offsets not loaded.";
        return false;
    }
    const auto i = static_cast<std::size_t>(cid);
    const std::uint64_t off = parent_louds_offsets_bytes_[i];
    const std::uint64_t end = parent_louds_offsets_bytes_[i + 1];
    if (end < off) {
        if (err) *err = "LinkageListReader::ReadClusterParentLOUDSBlob: invalid offsets.";
        return false;
    }
    const std::string path = JoinPath(dir_, "parent_louds.bin");
    return ReadBytesSpan(path, off, end - off, blob, err);
}

LinkageListThreadReader::~LinkageListThreadReader() {
    Close();
}

void LinkageListThreadReader::Close() {
    if (depth_offsets_in_.is_open()) depth_offsets_in_.close();
    if (real_ids_in_.is_open()) real_ids_in_.close();
    if (parent_in_.is_open()) parent_in_.close();
    if (parent_louds_in_.is_open()) parent_louds_in_.close();
    if (codes_in_.is_open()) codes_in_.close();
    if (coeffs_in_.is_open()) coeffs_in_.close();
    if (code0_one_in_.is_open()) code0_one_in_.close();
    if (a0_in_.is_open()) a0_in_.close();
    if (virt_codes_in_.is_open()) virt_codes_in_.close();
    if (virt_coeffs_in_.is_open()) virt_coeffs_in_.close();
    if (virt_a0_in_.is_open()) virt_a0_in_.close();
    src_ = nullptr;
    dir_.clear();
    meta_ = {};
    parent_louds_offsets_bytes_.clear();
}

bool LinkageListThreadReader::OpenFiles(std::string* err) {
    if (dir_.empty() || meta_.nlist <= 0 || meta_.m_codes < 0 ||
        meta_.small_code_width_bytes != LinkageListStoreConfig::kSmallCodeWidthBytes ||
        (meta_.code0_width_bytes != 1 && meta_.code0_width_bytes != 2 && meta_.code0_width_bytes != 4)) {
        if (err) *err = "LinkageListThreadReader::Open: invalid meta/dir.";
        return false;
    }
    depth_offsets_in_.open(JoinPath(dir_, "depth_offsets.u32"), std::ios::binary);
    real_ids_in_.open(JoinPath(dir_, "real_ids.u32"), std::ios::binary);
    if (meta_.store_parent_u32 != 0) {
        parent_in_.open(JoinPath(dir_, "parent.u32"), std::ios::binary);
        if (!parent_in_.is_open()) {
            if (meta_.store_parent_louds == 0) {
                if (err) *err = "LinkageListThreadReader::Open: failed to open parent.u32 and parent LOUDS is unavailable.";
                return false;
            }
            // Allow stores that omit (or have deleted) parent.u32 as long as parent LOUDS exists.
            meta_.store_parent_u32 = 0;
            static std::atomic<bool> warned{false};
            if (!warned.exchange(true)) {
                LogWarn("linkage_list: parent.u32 missing; will decode parent from parent_louds.bin on-demand.");
            }
        }
    }
    if (meta_.store_parent_louds != 0) {
        parent_louds_in_.open(JoinPath(dir_, "parent_louds.bin"), std::ios::binary);
    }
    codes_in_.open(JoinPath(dir_, "codes.bin"), std::ios::binary);
    code0_one_in_.open(JoinPath(dir_, "code0_one.bin"), std::ios::binary);
    virt_codes_in_.open(JoinPath(dir_, "virt_codes.bin"), std::ios::binary);
    if (meta_.store_coeffs_f32 != 0) {
        coeffs_in_.open(JoinPath(dir_, "coeffs.f32"), std::ios::binary);
        a0_in_.open(JoinPath(dir_, "a0.f32"), std::ios::binary);
        virt_coeffs_in_.open(JoinPath(dir_, "virt_coeffs.f32"), std::ios::binary);
        virt_a0_in_.open(JoinPath(dir_, "virt_a0.f32"), std::ios::binary);
        if (!require_coeffs_f32_ &&
            (!coeffs_in_.is_open() || !a0_in_.is_open() || !virt_coeffs_in_.is_open() || !virt_a0_in_.is_open())) {
            meta_.store_coeffs_f32 = 0;
            if (coeffs_in_.is_open()) coeffs_in_.close();
            if (a0_in_.is_open()) a0_in_.close();
            if (virt_coeffs_in_.is_open()) virt_coeffs_in_.close();
            if (virt_a0_in_.is_open()) virt_a0_in_.close();
            static std::atomic<bool> warned{false};
            if (!warned.exchange(true)) {
                LogWarn("linkage_list: float coeff payload missing; continuing without *.f32 payload files.");
            }
        }
    }
    if (!depth_offsets_in_.is_open() || !real_ids_in_.is_open() ||
        (meta_.store_parent_u32 != 0 && !parent_in_.is_open()) ||
        (meta_.store_parent_louds != 0 && !parent_louds_in_.is_open()) ||
        !codes_in_.is_open() || !code0_one_in_.is_open() || !virt_codes_in_.is_open() ||
        (meta_.store_coeffs_f32 != 0 &&
         (!coeffs_in_.is_open() || !a0_in_.is_open() || !virt_coeffs_in_.is_open() || !virt_a0_in_.is_open()))) {
        if (err) *err = "LinkageListThreadReader::Open: failed to open linkage_list payload files.";
        return false;
    }
    return true;
}

bool LinkageListThreadReader::OpenFrom(const LinkageListReader& src, std::string* err) {
    return OpenFrom(src, /*require_coeffs_f32=*/true, err);
}

bool LinkageListThreadReader::OpenFrom(const LinkageListReader& src,
                                     bool require_coeffs_f32,
                                     std::string* err) {
    Close();
    src_ = &src;
    dir_ = src.dir();
    meta_ = src.meta();
    require_coeffs_f32_ = require_coeffs_f32;
    parent_louds_offsets_bytes_.clear();
    if (meta_.store_parent_louds != 0) {
        parent_louds_offsets_bytes_ = src.parent_louds_offsets_bytes();
    }
    // If parent.u32 was not stored, avoid trying to open it.
    // For backward compatibility, we still allow meta_.store_parent_u32==1 but missing file: OpenFiles will downgrade.
    return OpenFiles(err);
}

bool LinkageListThreadReader::ReadAt(std::ifstream* in,
                                  std::uint64_t off,
                                  void* dst,
                                  std::size_t bytes,
                                  std::string* err) {
    if (bytes == 0) {
        return true;
    }
    in->seekg(static_cast<std::streamoff>(off), std::ios::beg);
    if (!(*in)) {
        if (err) *err = "LinkageListThreadReader: seek failed.";
        return false;
    }
    in->read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes));
    if (!(*in)) {
        if (err) *err = "LinkageListThreadReader: read failed.";
        return false;
    }
    return true;
}

bool LinkageListThreadReader::ReadClusterParentLOUDSBlob(int cid,
                                                       std::vector<std::uint8_t>* blob,
                                                       std::string* err) {
    if (!blob) {
        if (err) *err = "LinkageListThreadReader::ReadClusterParentLOUDSBlob: blob output is null.";
        return false;
    }
    blob->clear();
    if (!src_) {
        if (err) *err = "LinkageListThreadReader::ReadClusterParentLOUDSBlob: reader not open.";
        return false;
    }
    if (meta_.store_parent_louds == 0) {
        return true;
    }
    if (cid < 0 || cid >= meta_.nlist) {
        if (err) *err = "LinkageListThreadReader::ReadClusterParentLOUDSBlob: cid out of range.";
        return false;
    }
    if (parent_louds_offsets_bytes_.size() != static_cast<std::size_t>(meta_.nlist) + 1) {
        if (err) *err = "LinkageListThreadReader::ReadClusterParentLOUDSBlob: offsets not loaded.";
        return false;
    }
    const auto i = static_cast<std::size_t>(cid);
    const std::uint64_t off = parent_louds_offsets_bytes_[i];
    const std::uint64_t end = parent_louds_offsets_bytes_[i + 1];
    if (end < off) {
        if (err) *err = "LinkageListThreadReader::ReadClusterParentLOUDSBlob: invalid offsets.";
        return false;
    }
    const std::uint64_t bytes = end - off;
    blob->resize(static_cast<std::size_t>(bytes));
    if (bytes == 0) return true;
    return ReadAt(&parent_louds_in_, off, blob->data(), blob->size(), err);
}

bool LinkageListThreadReader::ReadCluster(int cid,
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
                                        std::string* err) {
    if (!src_) {
        if (err) *err = "LinkageListThreadReader::ReadCluster: reader not open.";
        return false;
    }
    std::uint64_t real_lo = 0, real_hi = 0, virt_lo = 0, virt_hi = 0, depth_lo = 0, depth_hi = 0;
    if (!src_->ReadClusterSpan(cid, &real_lo, &real_hi, &virt_lo, &virt_hi, &depth_lo, &depth_hi, err)) {
        return false;
    }
    const std::uint64_t n_real64 = real_hi - real_lo;
    const std::uint64_t n_virt64 = virt_hi - virt_lo;
    if (n_real64 > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) ||
        n_virt64 > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
        if (err) *err = "LinkageListThreadReader::ReadCluster: cluster too large.";
        return false;
    }
    const auto n_real = static_cast<std::uint32_t>(n_real64);
    const auto n_virt = static_cast<std::uint32_t>(n_virt64);
    const auto depth_len = static_cast<std::uint32_t>(depth_hi - depth_lo);

    if (real_ids_depth_order) {
        real_ids_depth_order->resize(n_real);
        if (!ReadAt(&real_ids_in_, real_lo * sizeof(std::uint32_t),
                    real_ids_depth_order->data(), n_real * sizeof(std::uint32_t), err)) return false;
    }
    if (parent_local_1based_depth_order) {
        if (meta_.store_parent_u32 != 0) {
            parent_local_1based_depth_order->resize(n_real);
            if (!ReadAt(&parent_in_, real_lo * sizeof(std::uint32_t),
                        parent_local_1based_depth_order->data(), n_real * sizeof(std::uint32_t), err)) return false;
        } else if (meta_.store_parent_louds != 0) {
            std::vector<std::uint8_t> blob;
            if (!ReadClusterParentLOUDSBlob(cid, &blob, err) || blob.empty()) {
                if (err && err->empty()) *err = "LinkageListThreadReader::ReadCluster: missing parent.u32 and failed to read parent LOUDS.";
                return false;
            }
            try {
                stlq::succinct::ParentLOUDS louds;
                louds.Deserialize(blob.data(),
                                  blob.size(),
                                  static_cast<std::uint32_t>(std::max(1, meta_.parent_louds_rank_words_per_super_log2)),
                                  /*build_indices=*/false);
                std::vector<std::uint32_t> parent_all;
                const std::size_t nc = static_cast<std::size_t>(n_real) + static_cast<std::size_t>(n_virt);
                louds.DecodeParent1Based(&parent_all, nc);
                parent_local_1based_depth_order->resize(n_real);
                for (std::size_t i = 0; i < static_cast<std::size_t>(n_real); ++i) {
                    (*parent_local_1based_depth_order)[i] = parent_all[static_cast<std::size_t>(n_virt) + i];
                }
                static std::atomic<bool> warned{false};
                if (!warned.exchange(true)) {
                    LogWarn("linkage_list: parent.u32 not stored; decoding parent from parent_louds.bin on-demand.");
                }
            } catch (...) {
                if (err && err->empty()) *err = "LinkageListThreadReader::ReadCluster: parent LOUDS decode failed.";
                return false;
            }
        } else {
            if (err) *err = "LinkageListThreadReader::ReadCluster: parent requested but neither parent.u32 nor parent LOUDS is available.";
            return false;
        }
    }
    if (depth_offsets) {
        depth_offsets->resize(depth_len);
        if (!ReadAt(&depth_offsets_in_, depth_lo * sizeof(std::uint32_t),
                    depth_offsets->data(), depth_len * sizeof(std::uint32_t), err)) return false;
    }

    const std::uint64_t codes_bytes =
        static_cast<std::uint64_t>(meta_.m_codes) * static_cast<std::uint64_t>(n_real) *
        static_cast<std::uint64_t>(meta_.small_code_width_bytes);
    const std::uint64_t codes_off =
        real_lo * static_cast<std::uint64_t>(meta_.m_codes) * static_cast<std::uint64_t>(meta_.small_code_width_bytes);
    if (codes_small_depth_order_bytes) {
        codes_small_depth_order_bytes->resize(static_cast<std::size_t>(codes_bytes));
        if (!ReadAt(&codes_in_, codes_off, codes_small_depth_order_bytes->data(),
                    static_cast<std::size_t>(codes_bytes), err)) return false;
    }

    if (meta_.store_coeffs_f32 != 0) {
        const std::uint64_t coeffs_count =
            static_cast<std::uint64_t>(meta_.m_codes) * static_cast<std::uint64_t>(n_real);
        const std::uint64_t coeffs_off = real_lo * static_cast<std::uint64_t>(meta_.m_codes) * sizeof(float);
        if (coeffs_small_depth_order) {
            coeffs_small_depth_order->resize(static_cast<std::size_t>(coeffs_count));
            if (!ReadAt(&coeffs_in_, coeffs_off, coeffs_small_depth_order->data(),
                        static_cast<std::size_t>(coeffs_count) * sizeof(float), err)) return false;
        }
    } else {
        if (coeffs_small_depth_order) coeffs_small_depth_order->clear();
    }

    const std::uint64_t code0_bytes = static_cast<std::uint64_t>(n_real) *
                                     static_cast<std::uint64_t>(meta_.code0_width_bytes);
    const std::uint64_t code0_off = real_lo * static_cast<std::uint64_t>(meta_.code0_width_bytes);
    if (code0_one_depth_order_bytes) {
        code0_one_depth_order_bytes->resize(static_cast<std::size_t>(code0_bytes));
        if (!ReadAt(&code0_one_in_, code0_off, code0_one_depth_order_bytes->data(),
                    static_cast<std::size_t>(code0_bytes), err)) return false;
    }

    if (meta_.store_coeffs_f32 != 0) {
        if (a0_depth_order) {
            a0_depth_order->resize(n_real);
            if (!ReadAt(&a0_in_, real_lo * sizeof(float), a0_depth_order->data(),
                        n_real * sizeof(float), err)) return false;
        }
    } else {
        if (a0_depth_order) a0_depth_order->clear();
    }

    const std::uint64_t vcodes_bytes =
        static_cast<std::uint64_t>(meta_.m_codes) * static_cast<std::uint64_t>(n_virt) *
        static_cast<std::uint64_t>(meta_.small_code_width_bytes);
    const std::uint64_t vcodes_off =
        virt_lo * static_cast<std::uint64_t>(meta_.m_codes) * static_cast<std::uint64_t>(meta_.small_code_width_bytes);
    if (virt_codes_small_bytes) {
        virt_codes_small_bytes->resize(static_cast<std::size_t>(vcodes_bytes));
        if (!ReadAt(&virt_codes_in_, vcodes_off, virt_codes_small_bytes->data(),
                    static_cast<std::size_t>(vcodes_bytes), err)) return false;
    }

    if (meta_.store_coeffs_f32 != 0) {
        const std::uint64_t vcoeffs_count =
            static_cast<std::uint64_t>(meta_.m_codes) * static_cast<std::uint64_t>(n_virt);
        const std::uint64_t vcoeffs_off = virt_lo * static_cast<std::uint64_t>(meta_.m_codes) * sizeof(float);
        if (virt_coeffs_small) {
            virt_coeffs_small->resize(static_cast<std::size_t>(vcoeffs_count));
            if (!ReadAt(&virt_coeffs_in_, vcoeffs_off, virt_coeffs_small->data(),
                        static_cast<std::size_t>(vcoeffs_count) * sizeof(float), err)) return false;
        }
        if (virt_a0) {
            virt_a0->resize(n_virt);
            if (!ReadAt(&virt_a0_in_, virt_lo * sizeof(float), virt_a0->data(),
                        n_virt * sizeof(float), err)) return false;
        }
    } else {
        if (virt_coeffs_small) virt_coeffs_small->clear();
        if (virt_a0) virt_a0->clear();
    }
    return true;
}

bool LinkageListRandomWriter::Open(const LinkageListStoreConfig& cfg,
                                 const LinkageListWritePlan& plan,
                                 std::string* err) {
    cfg_ = cfg;
    if (cfg_.dir.empty() || cfg_.nlist <= 0 || cfg_.m_codes < 0 ||
        (cfg_.code0_width_bytes != 1 && cfg_.code0_width_bytes != 2 && cfg_.code0_width_bytes != 4) ||
        (!cfg_.store_parent_u32 && !cfg_.store_parent_louds) ||
        (cfg_.store_parent_louds &&
         (cfg_.parent_louds_select_stride <= 0 ||
          cfg_.parent_louds_rank_words_per_super_log2 <= 0 ||
          cfg_.parent_louds_rank_words_per_super_log2 > 10))) {
        if (err) *err = "LinkageListRandomWriter::Open: invalid config.";
        return false;
    }
    if (plan.nlist != cfg_.nlist || plan.m_codes != cfg_.m_codes ||
        plan.small_code_width_bytes != LinkageListStoreConfig::kSmallCodeWidthBytes ||
        plan.code0_width_bytes != cfg_.code0_width_bytes) {
        if (err) *err = "LinkageListRandomWriter::Open: plan does not match config.";
        return false;
    }
    if (static_cast<int>(plan.real_offsets.size()) != cfg_.nlist + 1 ||
        static_cast<int>(plan.virt_offsets.size()) != cfg_.nlist + 1 ||
        static_cast<int>(plan.depth_offsets_offsets.size()) != cfg_.nlist + 1 ||
        (cfg_.store_parent_louds && static_cast<int>(plan.parent_louds_offsets_bytes.size()) != cfg_.nlist + 1)) {
        if (err) *err = "LinkageListRandomWriter::Open: invalid plan offsets.";
        return false;
    }
    if (!EnsureDir(cfg_.dir, err)) {
        return false;
    }

    // 1) Write offsets files (prefix sums) once.
    auto write_u64_vec = [&](const std::string& name, const std::vector<std::uint64_t>& v) -> bool {
        const std::string path = JoinPath(cfg_.dir, name);
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out.is_open()) {
            if (err) *err = "LinkageListRandomWriter::Open: failed to open " + path;
            return false;
        }
        out.write(reinterpret_cast<const char*>(v.data()),
                  static_cast<std::streamsize>(v.size() * sizeof(std::uint64_t)));
        if (!out) {
            if (err) *err = "LinkageListRandomWriter::Open: failed to write " + path;
            return false;
        }
        return true;
    };
    if (!write_u64_vec("real_offsets.u64", plan.real_offsets)) return false;
    if (!write_u64_vec("virt_offsets.u64", plan.virt_offsets)) return false;
    if (!write_u64_vec("depth_offsets_offsets.u64", plan.depth_offsets_offsets)) return false;
    if (cfg_.store_parent_louds) {
        if (!write_u64_vec("parent_louds_offsets.u64", plan.parent_louds_offsets_bytes)) return false;
    }

    // 2) Preallocate payload files to the expected final sizes.
    struct Prealloc {
        std::string name;
        std::uint64_t bytes = 0;
    };
    std::vector<Prealloc> files;
    files.reserve(cfg_.store_coeffs_f32 ? 12u : 8u);
    files.push_back({"depth_offsets.u32", plan.total_depth_offsets_u32_bytes});
    files.push_back({"real_ids.u32", plan.total_real_u32_bytes});
    if (cfg_.store_parent_u32) {
        files.push_back({"parent.u32", plan.total_parent_u32_bytes});
    }
    if (cfg_.store_parent_louds) {
        files.push_back({"parent_louds.bin", plan.total_parent_louds_bytes});
    }
    files.push_back({"codes.bin", plan.total_codes_bytes});
    files.push_back({"code0_one.bin", plan.total_code0_one_bytes});
    files.push_back({"virt_codes.bin", plan.total_virt_codes_bytes});
    if (cfg_.store_coeffs_f32) {
        files.push_back({"coeffs.f32", plan.total_coeffs_f32_bytes});
        files.push_back({"a0.f32", plan.total_a0_f32_bytes});
        files.push_back({"virt_coeffs.f32", plan.total_virt_coeffs_f32_bytes});
        files.push_back({"virt_a0.f32", plan.total_virt_a0_f32_bytes});
    }
    for (const auto& f : files) {
        const std::string path = JoinPath(cfg_.dir, f.name);
        // Create/truncate then resize. resize_file works on both Linux and Windows.
        {
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            if (!out.is_open()) {
                if (err) *err = "LinkageListRandomWriter::Open: failed to create " + path;
                return false;
            }
        }
        try {
            std::filesystem::resize_file(path, static_cast<std::uintmax_t>(f.bytes));
        } catch (...) {
            if (err) *err = "LinkageListRandomWriter::Open: failed to resize " + path;
            return false;
        }
    }

#if defined(_WIN32)
    // Windows: use fstream handles (seekp+write guarded externally inside WriteClusterAt).
    auto open_fstream = [&](std::fstream* f, const std::string& name) -> bool {
        const std::string path = JoinPath(cfg_.dir, name);
        f->open(path, std::ios::binary | std::ios::in | std::ios::out);
        if (!f->is_open()) {
            if (err) *err = "LinkageListRandomWriter::Open: failed to open " + path;
            return false;
        }
        return true;
    };
    if (!open_fstream(&depth_offsets_out_, "depth_offsets.u32")) return false;
    if (!open_fstream(&real_ids_out_, "real_ids.u32")) return false;
    if (cfg_.store_parent_u32) {
        if (!open_fstream(&parent_out_, "parent.u32")) return false;
    }
    if (cfg_.store_parent_louds) {
        if (!open_fstream(&parent_louds_out_, "parent_louds.bin")) return false;
    }
    if (!open_fstream(&codes_out_, "codes.bin")) return false;
    if (!open_fstream(&code0_one_out_, "code0_one.bin")) return false;
    if (!open_fstream(&virt_codes_out_, "virt_codes.bin")) return false;
    if (cfg_.store_coeffs_f32) {
        if (!open_fstream(&coeffs_out_, "coeffs.f32")) return false;
        if (!open_fstream(&a0_out_, "a0.f32")) return false;
        if (!open_fstream(&virt_coeffs_out_, "virt_coeffs.f32")) return false;
        if (!open_fstream(&virt_a0_out_, "virt_a0.f32")) return false;
    }
#else
    // Linux: open file descriptors for pwrite.
    fd_depth_offsets_ = OpenRwTrunc(JoinPath(cfg_.dir, "depth_offsets.u32"), err);
    if (fd_depth_offsets_ < 0) return false;
    fd_real_ids_ = OpenRwTrunc(JoinPath(cfg_.dir, "real_ids.u32"), err);
    if (fd_real_ids_ < 0) return false;
    if (cfg_.store_parent_u32) {
        fd_parent_ = OpenRwTrunc(JoinPath(cfg_.dir, "parent.u32"), err);
        if (fd_parent_ < 0) return false;
    }
    if (cfg_.store_parent_louds) {
        fd_parent_louds_ = OpenRwTrunc(JoinPath(cfg_.dir, "parent_louds.bin"), err);
        if (fd_parent_louds_ < 0) return false;
    }
    fd_codes_ = OpenRwTrunc(JoinPath(cfg_.dir, "codes.bin"), err);
    if (fd_codes_ < 0) return false;
    fd_code0_one_ = OpenRwTrunc(JoinPath(cfg_.dir, "code0_one.bin"), err);
    if (fd_code0_one_ < 0) return false;
    fd_virt_codes_ = OpenRwTrunc(JoinPath(cfg_.dir, "virt_codes.bin"), err);
    if (fd_virt_codes_ < 0) return false;
    if (cfg_.store_coeffs_f32) {
        fd_coeffs_ = OpenRwTrunc(JoinPath(cfg_.dir, "coeffs.f32"), err);
        if (fd_coeffs_ < 0) return false;
        fd_a0_ = OpenRwTrunc(JoinPath(cfg_.dir, "a0.f32"), err);
        if (fd_a0_ < 0) return false;
        fd_virt_coeffs_ = OpenRwTrunc(JoinPath(cfg_.dir, "virt_coeffs.f32"), err);
        if (fd_virt_coeffs_ < 0) return false;
        fd_virt_a0_ = OpenRwTrunc(JoinPath(cfg_.dir, "virt_a0.f32"), err);
        if (fd_virt_a0_ < 0) return false;
    }
#endif

    return true;
}

bool LinkageListRandomWriter::OpenResume(const LinkageListStoreConfig& cfg,
                                       const LinkageListWritePlan& plan,
                                       std::string* err) {
    cfg_ = cfg;
    if (cfg_.dir.empty() || cfg_.nlist <= 0 || cfg_.m_codes < 0 ||
        (cfg_.code0_width_bytes != 1 && cfg_.code0_width_bytes != 2 && cfg_.code0_width_bytes != 4) ||
        (!cfg_.store_parent_u32 && !cfg_.store_parent_louds) ||
        (cfg_.store_parent_louds &&
         (cfg_.parent_louds_select_stride <= 0 ||
          cfg_.parent_louds_rank_words_per_super_log2 <= 0 ||
          cfg_.parent_louds_rank_words_per_super_log2 > 10))) {
        if (err) *err = "LinkageListRandomWriter::OpenResume: invalid config.";
        return false;
    }
    if (plan.nlist != cfg_.nlist || plan.m_codes != cfg_.m_codes ||
        plan.small_code_width_bytes != LinkageListStoreConfig::kSmallCodeWidthBytes ||
        plan.code0_width_bytes != cfg_.code0_width_bytes) {
        if (err) *err = "LinkageListRandomWriter::OpenResume: plan does not match config.";
        return false;
    }
    if (static_cast<int>(plan.real_offsets.size()) != cfg_.nlist + 1 ||
        static_cast<int>(plan.virt_offsets.size()) != cfg_.nlist + 1 ||
        static_cast<int>(plan.depth_offsets_offsets.size()) != cfg_.nlist + 1 ||
        (cfg_.store_parent_louds && static_cast<int>(plan.parent_louds_offsets_bytes.size()) != cfg_.nlist + 1)) {
        if (err) *err = "LinkageListRandomWriter::OpenResume: invalid plan offsets.";
        return false;
    }
    if (!EnsureDir(cfg_.dir, err)) {
        return false;
    }

    // Re-write offsets files (small, deterministic) to match the provided plan.
    auto write_u64_vec = [&](const std::string& name, const std::vector<std::uint64_t>& v) -> bool {
        const std::string path = JoinPath(cfg_.dir, name);
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out.is_open()) {
            if (err) *err = "LinkageListRandomWriter::OpenResume: failed to open " + path;
            return false;
        }
        out.write(reinterpret_cast<const char*>(v.data()),
                  static_cast<std::streamsize>(v.size() * sizeof(std::uint64_t)));
        if (!out) {
            if (err) *err = "LinkageListRandomWriter::OpenResume: failed to write " + path;
            return false;
        }
        return true;
    };
    if (!write_u64_vec("real_offsets.u64", plan.real_offsets)) return false;
    if (!write_u64_vec("virt_offsets.u64", plan.virt_offsets)) return false;
    if (!write_u64_vec("depth_offsets_offsets.u64", plan.depth_offsets_offsets)) return false;
    if (cfg_.store_parent_louds) {
        if (!write_u64_vec("parent_louds_offsets.u64", plan.parent_louds_offsets_bytes)) return false;
    }

    // Validate that the payload files already exist at the expected final sizes.
    struct Want {
        std::string name;
        std::uint64_t bytes = 0;
    };
    std::vector<Want> files;
    files.reserve(cfg_.store_coeffs_f32 ? 12u : 8u);
    files.push_back({"depth_offsets.u32", plan.total_depth_offsets_u32_bytes});
    files.push_back({"real_ids.u32", plan.total_real_u32_bytes});
    if (cfg_.store_parent_u32) {
        files.push_back({"parent.u32", plan.total_parent_u32_bytes});
    }
    if (cfg_.store_parent_louds) {
        files.push_back({"parent_louds.bin", plan.total_parent_louds_bytes});
    }
    files.push_back({"codes.bin", plan.total_codes_bytes});
    files.push_back({"code0_one.bin", plan.total_code0_one_bytes});
    files.push_back({"virt_codes.bin", plan.total_virt_codes_bytes});
    if (cfg_.store_coeffs_f32) {
        files.push_back({"coeffs.f32", plan.total_coeffs_f32_bytes});
        files.push_back({"a0.f32", plan.total_a0_f32_bytes});
        files.push_back({"virt_coeffs.f32", plan.total_virt_coeffs_f32_bytes});
        files.push_back({"virt_a0.f32", plan.total_virt_a0_f32_bytes});
    }
    for (const auto& f : files) {
        const std::string path = JoinPath(cfg_.dir, f.name);
        std::error_code ec;
        if (!std::filesystem::exists(path, ec)) {
            if (err) *err = "LinkageListRandomWriter::OpenResume: missing payload file: " + path;
            return false;
        }
        const std::uint64_t got = static_cast<std::uint64_t>(std::filesystem::file_size(path, ec));
        if (ec) {
            if (err) *err = "LinkageListRandomWriter::OpenResume: failed to stat payload file: " + path;
            return false;
        }
        if (got != f.bytes) {
            if (err) {
                *err = "LinkageListRandomWriter::OpenResume: payload file size mismatch: " + path +
                       " got=" + std::to_string(got) + " want=" + std::to_string(f.bytes);
            }
            return false;
        }
    }

#if defined(_WIN32)
    auto open_fstream = [&](std::fstream* f, const std::string& name) -> bool {
        const std::string path = JoinPath(cfg_.dir, name);
        f->open(path, std::ios::binary | std::ios::in | std::ios::out);
        if (!f->is_open()) {
            if (err) *err = "LinkageListRandomWriter::OpenResume: failed to open " + path;
            return false;
        }
        return true;
    };
    if (!open_fstream(&depth_offsets_out_, "depth_offsets.u32")) return false;
    if (!open_fstream(&real_ids_out_, "real_ids.u32")) return false;
    if (cfg_.store_parent_u32) {
        if (!open_fstream(&parent_out_, "parent.u32")) return false;
    }
    if (cfg_.store_parent_louds) {
        if (!open_fstream(&parent_louds_out_, "parent_louds.bin")) return false;
    }
    if (!open_fstream(&codes_out_, "codes.bin")) return false;
    if (!open_fstream(&code0_one_out_, "code0_one.bin")) return false;
    if (!open_fstream(&virt_codes_out_, "virt_codes.bin")) return false;
    if (cfg_.store_coeffs_f32) {
        if (!open_fstream(&coeffs_out_, "coeffs.f32")) return false;
        if (!open_fstream(&a0_out_, "a0.f32")) return false;
        if (!open_fstream(&virt_coeffs_out_, "virt_coeffs.f32")) return false;
        if (!open_fstream(&virt_a0_out_, "virt_a0.f32")) return false;
    }
#else
    fd_depth_offsets_ = OpenRwNoTrunc(JoinPath(cfg_.dir, "depth_offsets.u32"), err);
    if (fd_depth_offsets_ < 0) return false;
    fd_real_ids_ = OpenRwNoTrunc(JoinPath(cfg_.dir, "real_ids.u32"), err);
    if (fd_real_ids_ < 0) return false;
    if (cfg_.store_parent_u32) {
        fd_parent_ = OpenRwNoTrunc(JoinPath(cfg_.dir, "parent.u32"), err);
        if (fd_parent_ < 0) return false;
    }
    if (cfg_.store_parent_louds) {
        fd_parent_louds_ = OpenRwNoTrunc(JoinPath(cfg_.dir, "parent_louds.bin"), err);
        if (fd_parent_louds_ < 0) return false;
    }
    fd_codes_ = OpenRwNoTrunc(JoinPath(cfg_.dir, "codes.bin"), err);
    if (fd_codes_ < 0) return false;
    fd_code0_one_ = OpenRwNoTrunc(JoinPath(cfg_.dir, "code0_one.bin"), err);
    if (fd_code0_one_ < 0) return false;
    fd_virt_codes_ = OpenRwNoTrunc(JoinPath(cfg_.dir, "virt_codes.bin"), err);
    if (fd_virt_codes_ < 0) return false;
    if (cfg_.store_coeffs_f32) {
        fd_coeffs_ = OpenRwNoTrunc(JoinPath(cfg_.dir, "coeffs.f32"), err);
        if (fd_coeffs_ < 0) return false;
        fd_a0_ = OpenRwNoTrunc(JoinPath(cfg_.dir, "a0.f32"), err);
        if (fd_a0_ < 0) return false;
        fd_virt_coeffs_ = OpenRwNoTrunc(JoinPath(cfg_.dir, "virt_coeffs.f32"), err);
        if (fd_virt_coeffs_ < 0) return false;
        fd_virt_a0_ = OpenRwNoTrunc(JoinPath(cfg_.dir, "virt_a0.f32"), err);
        if (fd_virt_a0_ < 0) return false;
    }
#endif

    return true;
}

bool LinkageListRandomWriter::WriteClusterAt(int cid,
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
                                          std::string* err) {
    if (cid < 0 || cid >= plan.nlist) {
        if (err) *err = "LinkageListRandomWriter::WriteClusterAt: cid out of range.";
        return false;
    }
    const auto i = static_cast<std::size_t>(cid);
    const std::uint32_t n_real = plan.n_real[i];
    const std::uint32_t n_virt = plan.n_virt[i];
    const std::uint32_t depth_len = plan.depth_len[i];
    if (real_ids_depth_order.size() != n_real ||
        parent_local_1based_depth_order.size() != n_real ||
        depth_offsets.size() != depth_len ||
        (cfg_.store_coeffs_f32 && coeffs_small_depth_order.size() != static_cast<std::size_t>(plan.m_codes) * n_real) ||
        (cfg_.store_coeffs_f32 && a0_depth_order.size() != n_real) ||
        code0_one_depth_order_bytes.size() != static_cast<std::size_t>(n_real) * plan.code0_width_bytes ||
        (cfg_.store_coeffs_f32 && virt_coeffs_small.size() != static_cast<std::size_t>(plan.m_codes) * n_virt) ||
        virt_codes_small_bytes.size() != static_cast<std::size_t>(plan.m_codes) * n_virt * plan.small_code_width_bytes ||
        (cfg_.store_coeffs_f32 && virt_a0.size() != n_virt)) {
        if (err) *err = "LinkageListRandomWriter::WriteClusterAt: payload size mismatch.";
        return false;
    }
    if (codes_small_depth_order_bytes.size() !=
        static_cast<std::size_t>(plan.m_codes) * n_real * plan.small_code_width_bytes) {
        if (err) *err = "LinkageListRandomWriter::WriteClusterAt: codes size mismatch.";
        return false;
    }

    const std::uint64_t real_lo = plan.real_offsets[i];
    const std::uint64_t virt_lo = plan.virt_offsets[i];
    const std::uint64_t depth_lo = plan.depth_offsets_offsets[i];

    const std::uint64_t off_depth_bytes = depth_lo * sizeof(std::uint32_t);
    const std::uint64_t off_real_u32 = real_lo * sizeof(std::uint32_t);
    const std::uint64_t off_parent_u32 = real_lo * sizeof(std::uint32_t);
    const std::uint64_t off_codes = real_lo * static_cast<std::uint64_t>(plan.m_codes) *
                                    static_cast<std::uint64_t>(plan.small_code_width_bytes);
    const std::uint64_t off_coeffs = real_lo * static_cast<std::uint64_t>(plan.m_codes) * sizeof(float);
    const std::uint64_t off_code0 = real_lo * static_cast<std::uint64_t>(plan.code0_width_bytes);
    const std::uint64_t off_a0 = real_lo * sizeof(float);

    const std::uint64_t off_vcodes = virt_lo * static_cast<std::uint64_t>(plan.m_codes) *
                                     static_cast<std::uint64_t>(plan.small_code_width_bytes);
    const std::uint64_t off_vcoeffs = virt_lo * static_cast<std::uint64_t>(plan.m_codes) * sizeof(float);
    const std::uint64_t off_va0 = virt_lo * sizeof(float);

    std::uint64_t off_parent_louds = 0;
    std::uint64_t parent_louds_bytes = 0;
    if (cfg_.store_parent_louds) {
        off_parent_louds = plan.parent_louds_offsets_bytes[i];
        const std::uint64_t end = plan.parent_louds_offsets_bytes[i + 1];
        if (end < off_parent_louds) {
            if (err) *err = "LinkageListRandomWriter::WriteClusterAt: invalid parent_louds offsets.";
            return false;
        }
        parent_louds_bytes = end - off_parent_louds;
    }

#if defined(_WIN32)
    static std::mutex mu_depth, mu_real, mu_parent, mu_parent_louds,
        mu_codes, mu_coeffs, mu_code0, mu_a0, mu_vcodes, mu_vcoeffs, mu_va0;
    auto seek_write = [&](std::fstream* f, std::mutex* mu, std::uint64_t off, const void* buf, std::size_t bytes) -> bool {
        if (bytes == 0) return true;
        std::lock_guard<std::mutex> lock(*mu);
        f->seekp(static_cast<std::streamoff>(off), std::ios::beg);
        if (!(*f)) return false;
        f->write(reinterpret_cast<const char*>(buf), static_cast<std::streamsize>(bytes));
        return static_cast<bool>(*f);
    };
    if (!seek_write(&depth_offsets_out_, &mu_depth, off_depth_bytes, depth_offsets.data(),
                    depth_offsets.size() * sizeof(std::uint32_t))) return false;
    if (!seek_write(&real_ids_out_, &mu_real, off_real_u32, real_ids_depth_order.data(),
                    real_ids_depth_order.size() * sizeof(std::uint32_t))) return false;
    if (cfg_.store_parent_u32) {
        if (!seek_write(&parent_out_, &mu_parent, off_parent_u32, parent_local_1based_depth_order.data(),
                        parent_local_1based_depth_order.size() * sizeof(std::uint32_t))) return false;
    }
    if (cfg_.store_parent_louds) {
        const std::size_t n_total = static_cast<std::size_t>(n_real + n_virt);
        std::vector<std::uint32_t> parent_all(n_total, 0u);
        const std::size_t n_virt_sz = static_cast<std::size_t>(n_virt);
        const std::size_t n_real_sz = static_cast<std::size_t>(n_real);
        std::memcpy(parent_all.data() + n_virt_sz,
                    parent_local_1based_depth_order.data(),
                    n_real_sz * sizeof(std::uint32_t));
        stlq::succinct::ParentLOUDS louds;
        louds.BuildFromParent1Based(parent_all.data(),
                                    parent_all.size(),
                                    static_cast<std::uint32_t>(std::max(1, cfg_.parent_louds_select_stride)),
                                    static_cast<std::uint32_t>(std::max(1, cfg_.parent_louds_rank_words_per_super_log2)),
                                    /*build_indices=*/false);
        const std::vector<std::uint8_t> blob = louds.Serialize();
        if (blob.size() != static_cast<std::size_t>(parent_louds_bytes)) {
            if (err) *err = "LinkageListRandomWriter::WriteClusterAt: parent_louds blob size mismatch.";
            return false;
        }
        if (!seek_write(&parent_louds_out_, &mu_parent_louds, off_parent_louds,
                        blob.data(), blob.size())) return false;
    }
    if (!seek_write(&codes_out_, &mu_codes, off_codes, codes_small_depth_order_bytes.data(),
                    codes_small_depth_order_bytes.size())) return false;
    if (cfg_.store_coeffs_f32) {
        if (!seek_write(&coeffs_out_, &mu_coeffs, off_coeffs, coeffs_small_depth_order.data(),
                        coeffs_small_depth_order.size() * sizeof(float))) return false;
    }
    if (!seek_write(&code0_one_out_, &mu_code0, off_code0, code0_one_depth_order_bytes.data(),
                    code0_one_depth_order_bytes.size())) return false;
    if (cfg_.store_coeffs_f32) {
        if (!seek_write(&a0_out_, &mu_a0, off_a0, a0_depth_order.data(),
                        a0_depth_order.size() * sizeof(float))) return false;
    }
    if (!seek_write(&virt_codes_out_, &mu_vcodes, off_vcodes, virt_codes_small_bytes.data(),
                    virt_codes_small_bytes.size())) return false;
    if (cfg_.store_coeffs_f32) {
        if (!seek_write(&virt_coeffs_out_, &mu_vcoeffs, off_vcoeffs, virt_coeffs_small.data(),
                        virt_coeffs_small.size() * sizeof(float))) return false;
        if (!seek_write(&virt_a0_out_, &mu_va0, off_va0, virt_a0.data(),
                        virt_a0.size() * sizeof(float))) return false;
    }
#else
    if (!PwriteAll(fd_depth_offsets_, depth_offsets.data(),
                   depth_offsets.size() * sizeof(std::uint32_t), off_depth_bytes, err)) return false;
    if (!PwriteAll(fd_real_ids_, real_ids_depth_order.data(),
                   real_ids_depth_order.size() * sizeof(std::uint32_t), off_real_u32, err)) return false;
    if (cfg_.store_parent_u32) {
        if (!PwriteAll(fd_parent_, parent_local_1based_depth_order.data(),
                       parent_local_1based_depth_order.size() * sizeof(std::uint32_t), off_parent_u32, err)) return false;
    }
    if (cfg_.store_parent_louds) {
        const auto n_total = static_cast<std::size_t>(n_real + n_virt);
        std::vector<std::uint32_t> parent_all(n_total, 0u);
        const auto n_virt_sz = static_cast<std::size_t>(n_virt);
        const auto n_real_sz = static_cast<std::size_t>(n_real);
        std::memcpy(parent_all.data() + n_virt_sz,
                    parent_local_1based_depth_order.data(),
                    n_real_sz * sizeof(std::uint32_t));
        stlq::succinct::ParentLOUDS louds;
        louds.BuildFromParent1Based(parent_all.data(),
                                    parent_all.size(),
                                    static_cast<std::uint32_t>(std::max(1, cfg_.parent_louds_select_stride)),
                                    static_cast<std::uint32_t>(std::max(1, cfg_.parent_louds_rank_words_per_super_log2)),
                                    /*build_indices=*/false);
        const std::vector<std::uint8_t> blob = louds.Serialize();
        if (blob.size() != static_cast<std::size_t>(parent_louds_bytes)) {
            if (err) *err = "LinkageListRandomWriter::WriteClusterAt: parent_louds blob size mismatch.";
            return false;
        }
        if (!PwriteAll(fd_parent_louds_, blob.data(), blob.size(), off_parent_louds, err)) return false;
    }
    if (!PwriteAll(fd_codes_, codes_small_depth_order_bytes.data(),
                   codes_small_depth_order_bytes.size(), off_codes, err)) return false;
    if (cfg_.store_coeffs_f32) {
        if (!PwriteAll(fd_coeffs_, coeffs_small_depth_order.data(),
                       coeffs_small_depth_order.size() * sizeof(float), off_coeffs, err)) return false;
    }
    if (!PwriteAll(fd_code0_one_, code0_one_depth_order_bytes.data(),
                   code0_one_depth_order_bytes.size(), off_code0, err)) return false;
    if (cfg_.store_coeffs_f32) {
        if (!PwriteAll(fd_a0_, a0_depth_order.data(),
                       a0_depth_order.size() * sizeof(float), off_a0, err)) return false;
    }
    if (!PwriteAll(fd_virt_codes_, virt_codes_small_bytes.data(),
                   virt_codes_small_bytes.size(), off_vcodes, err)) return false;
    if (cfg_.store_coeffs_f32) {
        if (!PwriteAll(fd_virt_coeffs_, virt_coeffs_small.data(),
                       virt_coeffs_small.size() * sizeof(float), off_vcoeffs, err)) return false;
        if (!PwriteAll(fd_virt_a0_, virt_a0.data(),
                       virt_a0.size() * sizeof(float), off_va0, err)) return false;
    }
#endif

    return true;
}

bool LinkageListRandomWriter::WriteMeta(std::string* err) const {
    struct Header {
        char magic[8]{};
        std::uint32_t version = 4;
        std::uint32_t nlist = 0;
        std::uint32_t m_codes = 0;
        std::uint32_t small_code_width_bytes = 0;
        std::uint32_t code0_width_bytes = 0;
        std::uint32_t reserved[8] = {0};
    } h;
    std::memcpy(h.magic, "LINKAGELST", 8);
    h.nlist = static_cast<std::uint32_t>(cfg_.nlist);
    h.m_codes = static_cast<std::uint32_t>(cfg_.m_codes);
    h.small_code_width_bytes = static_cast<std::uint32_t>(LinkageListStoreConfig::kSmallCodeWidthBytes);
    h.code0_width_bytes = static_cast<std::uint32_t>(cfg_.code0_width_bytes);
    h.reserved[0] = cfg_.store_coeffs_f32 ? 1u : 0u;
    h.reserved[4] = cfg_.store_parent_u32 ? 1u : 0u;
    h.reserved[1] = cfg_.store_parent_louds ? 1u : 0u;
    h.reserved[2] = static_cast<std::uint32_t>(std::max(1, cfg_.parent_louds_select_stride));
    h.reserved[3] = static_cast<std::uint32_t>(std::max(1, cfg_.parent_louds_rank_words_per_super_log2));

    const std::string path = JoinPath(cfg_.dir, "meta.bin");
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        if (err) *err = "LinkageListRandomWriter::Finish: failed to open meta.bin.";
        return false;
    }
    out.write(reinterpret_cast<const char*>(&h), sizeof(Header));
    if (!out) {
        if (err) *err = "LinkageListRandomWriter::Finish: failed to write meta.bin.";
        return false;
    }
    return true;
}

bool LinkageListRandomWriter::Finish(std::string* err) {
#if defined(_WIN32)
    depth_offsets_out_.close();
    real_ids_out_.close();
    if (cfg_.store_parent_u32) {
        parent_out_.close();
    }
    if (cfg_.store_parent_louds) {
        parent_louds_out_.close();
    }
    codes_out_.close();
    code0_one_out_.close();
    virt_codes_out_.close();
    if (cfg_.store_coeffs_f32) {
        coeffs_out_.close();
        a0_out_.close();
        virt_coeffs_out_.close();
        virt_a0_out_.close();
    }
#else
    CloseFd(&fd_depth_offsets_);
    CloseFd(&fd_real_ids_);
    if (cfg_.store_parent_u32) {
        CloseFd(&fd_parent_);
    }
    if (cfg_.store_parent_louds) {
        CloseFd(&fd_parent_louds_);
    }
    CloseFd(&fd_codes_);
    CloseFd(&fd_code0_one_);
    CloseFd(&fd_virt_codes_);
    if (cfg_.store_coeffs_f32) {
        CloseFd(&fd_coeffs_);
        CloseFd(&fd_a0_);
        CloseFd(&fd_virt_coeffs_);
        CloseFd(&fd_virt_a0_);
    }
#endif
    return WriteMeta(err);
}

}  // namespace stlq::io

namespace stlq::io {

bool RewriteLinkageListMeta(const std::string& dir,
                          const LinkageListMeta& meta,
                          std::string* err) {
    if (dir.empty() || meta.nlist <= 0 || meta.m_codes < 0 ||
        meta.small_code_width_bytes != LinkageListStoreConfig::kSmallCodeWidthBytes ||
        (meta.code0_width_bytes != 1 && meta.code0_width_bytes != 2 && meta.code0_width_bytes != 4)) {
        if (err) *err = "RewriteLinkageListMeta: invalid dir/meta.";
        return false;
    }
    struct Header {
        char magic[8]{};
        std::uint32_t version = 4;
        std::uint32_t nlist = 0;
        std::uint32_t m_codes = 0;
        std::uint32_t small_code_width_bytes = 0;
        std::uint32_t code0_width_bytes = 0;
        std::uint32_t reserved[8] = {0};
    } h;
    std::memcpy(h.magic, "LINKAGELST", 8);
    h.nlist = static_cast<std::uint32_t>(meta.nlist);
    h.m_codes = static_cast<std::uint32_t>(meta.m_codes);
    h.small_code_width_bytes = static_cast<std::uint32_t>(meta.small_code_width_bytes);
    h.code0_width_bytes = static_cast<std::uint32_t>(meta.code0_width_bytes);
    h.reserved[0] = (meta.store_coeffs_f32 != 0) ? 1u : 0u;
    h.reserved[1] = (meta.store_parent_louds != 0) ? 1u : 0u;
    h.reserved[2] = static_cast<std::uint32_t>(std::max(1, meta.parent_louds_select_stride));
    h.reserved[3] = static_cast<std::uint32_t>(std::max(1, meta.parent_louds_rank_words_per_super_log2));
    h.reserved[4] = (meta.store_parent_u32 != 0) ? 1u : 0u;

    const std::string path = (std::filesystem::path(dir) / "meta.bin").string();
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        if (err) *err = "RewriteLinkageListMeta: failed to open meta.bin.";
        return false;
    }
    out.write(reinterpret_cast<const char*>(&h), sizeof(Header));
    if (!out) {
        if (err) *err = "RewriteLinkageListMeta: failed to write meta.bin.";
        return false;
    }
    return true;
}

bool UpdateLinkageListMetaStoreCoeffsF32(const std::string& dir,
                                       bool store_coeffs_f32,
                                       std::string* err) {
    LinkageListReader reader;
    std::string local_err;
    if (!reader.Open(dir, &local_err)) {
        if (err) *err = local_err.empty() ? "UpdateLinkageListMetaStoreCoeffsF32: failed to open linkage_list." : local_err;
        return false;
    }
    LinkageListMeta meta = reader.meta();
    meta.store_coeffs_f32 = store_coeffs_f32 ? 1 : 0;
    return RewriteLinkageListMeta(dir, meta, err);
}

}  // namespace stlq::io
