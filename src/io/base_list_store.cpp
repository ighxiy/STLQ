#include "stlq/io/base_list_store.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "stlq/pipeline/app_utils.h"
#include "stlq/common/logger.h"
#include "stlq/io/base_store.h"
#include "stlq/io/ivf_lists.h"
#include "stlq/common/timer.h"

#if !defined(_WIN32)
#include <cerrno>
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

std::string ZeroPad(int v, int width) {
    std::string s = std::to_string(v);
    if (static_cast<int>(s.size()) >= width) return s;
    return std::string(static_cast<std::size_t>(width - s.size()), '0') + s;
}

bool EnsureDir(const std::string& dir, std::string* err) {
    try {
        std::filesystem::create_directories(dir);
        return true;
    } catch (...) {
        if (err) *err = "BaseListStore: failed to create dir: " + dir;
        return false;
    }
}

bool ReadAll(std::ifstream& in, void* dst, std::size_t bytes) {
    in.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes));
    return static_cast<std::size_t>(in.gcount()) == bytes;
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
                *err = "BuildBaseListStoreFromBasicBuckets: pwrite failed: " +
                       std::string(std::strerror(errno));
            }
            return false;
        }
        if (w == 0) {
            if (err) *err = "BuildBaseListStoreFromBasicBuckets: pwrite returned 0.";
            return false;
        }
        remaining -= static_cast<std::size_t>(w);
        p += w;
        cur += static_cast<std::uint64_t>(w);
    }
    return true;
}

int OpenRw(const std::string& path, std::string* err) {
    const int fd = ::open(path.c_str(), O_CREAT | O_RDWR, 0644);
    if (fd < 0) {
        if (err) {
            *err = "BuildBaseListStoreFromBasicBuckets: open failed: " + path + " : " +
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

#if defined(_WIN32)
bool SeekWriteAll(std::fstream* f,
                  std::uint64_t off,
                  const void* buf,
                  std::size_t bytes,
                  std::string* err) {
    f->seekp(static_cast<std::streamoff>(off), std::ios::beg);
    if (!(*f)) {
        if (err) *err = "BuildBaseListStoreFromBasicBuckets: seekp failed.";
        return false;
    }
    if (bytes == 0) return true;
    f->write(reinterpret_cast<const char*>(buf), static_cast<std::streamsize>(bytes));
    if (!(*f)) {
        if (err) *err = "BuildBaseListStoreFromBasicBuckets: write failed.";
        return false;
    }
    return true;
}
#endif

}  // namespace

bool BaseListReader::Open(const std::string& dir, std::string* err) {
    dir_ = dir;
    meta_ = {};
    if (dir_.empty()) {
        if (err) *err = "BaseListReader::Open: empty dir.";
        return false;
    }

    const std::string meta_path = JoinPath(dir_, "meta.bin");
    std::ifstream in(meta_path, std::ios::binary);
    if (!in.is_open()) {
        if (err) *err = "BaseListReader::Open: failed to open " + meta_path;
        return false;
    }
    struct Header {
        std::uint32_t magic = 0;
        std::uint32_t version = 0;
        std::int32_t d = 0;
        std::int32_t m = 0;
        std::int32_t m_codes = 0;
        std::uint32_t reserved = 0;
        std::uint64_t ntotal = 0;
    } h;
    if (!ReadAll(in, &h, sizeof(Header))) {
        if (err) *err = "BaseListReader::Open: failed to read meta header.";
        return false;
    }
    if (h.magic != 0x534C5342u || h.version != 1) {  // "BSLS"
        if (err) *err = "BaseListReader::Open: unsupported meta.bin version.";
        return false;
    }
    if (h.d <= 0 || h.m <= 0 || h.m_codes < 0) {
        if (err) *err = "BaseListReader::Open: invalid meta fields.";
        return false;
    }
    meta_.d = h.d;
    meta_.m = h.m;
    meta_.m_codes = h.m_codes;
    meta_.ntotal = h.ntotal;

    // Optional raw vectors aligned with list-order.
    // Presence is detected by file existence (meta.bin remains versioned for core fields only).
    meta_.has_raw_u8 = false;
    meta_.raw_u8_bytes_per_vec = 0;
    meta_.has_raw_f32 = false;
    meta_.raw_f32_bytes_per_vec = 0;
    const std::string raw_path = JoinPath(dir_, "raw_u8.bin");
    std::error_code ec;
    if (std::filesystem::exists(raw_path, ec)) {
        meta_.has_raw_u8 = true;
        meta_.raw_u8_bytes_per_vec = meta_.d;
    }
    const std::string raw_f32_path = JoinPath(dir_, "raw_f32.bin");
    if (std::filesystem::exists(raw_f32_path, ec)) {
        meta_.has_raw_f32 = true;
        meta_.raw_f32_bytes_per_vec = meta_.d * static_cast<int>(sizeof(float));
    }
    return true;
}

bool BaseListReader::ReadCodesSpan(std::uint64_t begin,
                                  std::uint32_t count,
                                  std::vector<Code>* out,
                                  std::string* err) const {
    if (!out) {
        if (err) *err = "BaseListReader::ReadCodesSpan: null output.";
        return false;
    }
    out->clear();
    if (count == 0 || meta_.m_codes == 0) {
        return true;
    }
    if (meta_.ntotal > 0 && begin >= meta_.ntotal) {
        return true;
    }
    const std::string path = JoinPath(dir_, "codes.bin");
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        if (err) *err = "BaseListReader::ReadCodesSpan: failed to open " + path;
        return false;
    }
    const auto rec = static_cast<std::uint64_t>(meta_.m_codes);
    const std::uint64_t off = begin * rec;
    in.seekg(static_cast<std::streamoff>(off), std::ios::beg);
    if (!in) {
        if (err) *err = "BaseListReader::ReadCodesSpan: seek failed.";
        return false;
    }
    const std::size_t bytes = static_cast<std::size_t>(rec) * static_cast<std::size_t>(count);
    out->resize(bytes);
    in.read(reinterpret_cast<char*>(out->data()), static_cast<std::streamsize>(bytes));
    if (!in) {
        if (err) *err = "BaseListReader::ReadCodesSpan: read failed.";
        return false;
    }
    return true;
}

bool BaseListReader::ReadCoeffsSpan(std::uint64_t begin,
                                   std::uint32_t count,
                                   std::vector<float>* out,
                                   std::string* err) const {
    if (!out) {
        if (err) *err = "BaseListReader::ReadCoeffsSpan: null output.";
        return false;
    }
    out->clear();
    if (count == 0) {
        return true;
    }
    if (meta_.ntotal > 0 && begin >= meta_.ntotal) {
        return true;
    }
    const std::string path = JoinPath(dir_, "coeffs.bin");
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        if (err) *err = "BaseListReader::ReadCoeffsSpan: failed to open " + path;
        return false;
    }
    const std::uint64_t rec = static_cast<std::uint64_t>(meta_.m) * sizeof(float);
    const std::uint64_t off = begin * rec;
    in.seekg(static_cast<std::streamoff>(off), std::ios::beg);
    if (!in) {
        if (err) *err = "BaseListReader::ReadCoeffsSpan: seek failed.";
        return false;
    }
    const std::size_t bytes = static_cast<std::size_t>(meta_.m) * static_cast<std::size_t>(count);
    out->resize(bytes);
    in.read(reinterpret_cast<char*>(out->data()), static_cast<std::streamsize>(bytes * sizeof(float)));
    if (!in) {
        if (err) *err = "BaseListReader::ReadCoeffsSpan: read failed.";
        return false;
    }
    return true;
}

bool BaseListReader::ReadRawU8Span(std::uint64_t begin,
                                  std::uint32_t count,
                                  ColMajorMatrix<std::uint8_t>* out,
                                  std::string* err) const {
    if (!out) {
        if (err) *err = "BaseListReader::ReadRawU8Span: null output.";
        return false;
    }
    out->rows = 0;
    out->cols = 0;
    out->data.clear();
    if (!meta_.has_raw_u8) {
        if (err) *err = "BaseListReader::ReadRawU8Span: raw_u8.bin is not available.";
        return false;
    }
    if (count == 0) {
        return true;
    }
    if (meta_.ntotal > 0 && begin >= meta_.ntotal) {
        return true;
    }

    const std::string path = JoinPath(dir_, "raw_u8.bin");
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        if (err) *err = "BaseListReader::ReadRawU8Span: failed to open " + path;
        return false;
    }

    const auto rec = static_cast<std::uint64_t>(meta_.raw_u8_bytes_per_vec);
    const std::uint64_t off = begin * rec;
    in.seekg(static_cast<std::streamoff>(off), std::ios::beg);
    if (!in) {
        if (err) *err = "BaseListReader::ReadRawU8Span: seek failed.";
        return false;
    }

    const std::size_t bytes = static_cast<std::size_t>(rec) * static_cast<std::size_t>(count);
    out->rows = meta_.d;
    out->cols = static_cast<int>(count);
    out->data.resize(bytes);
    in.read(reinterpret_cast<char*>(out->data.data()), static_cast<std::streamsize>(bytes));
    if (!in) {
        if (err) *err = "BaseListReader::ReadRawU8Span: read failed.";
        return false;
    }
    return true;
}

bool BaseListReader::ReadRawF32Span(std::uint64_t begin,
                                   std::uint32_t count,
                                   ColMajorMatrix<float>* out,
                                   std::string* err) const {
    if (!out) {
        if (err) *err = "BaseListReader::ReadRawF32Span: null output.";
        return false;
    }
    out->rows = meta_.d;
    out->cols = 0;
    out->data.clear();
    if (count == 0) {
        return true;
    }
    if (!meta_.has_raw_f32) {
        if (err) *err = "BaseListReader::ReadRawF32Span: raw_f32.bin is not available.";
        return false;
    }
    if (meta_.ntotal > 0 && begin >= meta_.ntotal) {
        return true;
    }

    const std::string path = JoinPath(dir_, "raw_f32.bin");
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        if (err) *err = "BaseListReader::ReadRawF32Span: failed to open " + path;
        return false;
    }

    const std::uint64_t rec = static_cast<std::uint64_t>(meta_.d) * sizeof(float);
    const std::uint64_t off = begin * rec;
    in.seekg(static_cast<std::streamoff>(off), std::ios::beg);
    if (!in) {
        if (err) *err = "BaseListReader::ReadRawF32Span: seek failed.";
        return false;
    }

    out->cols = static_cast<int>(count);
    out->data.resize(static_cast<std::size_t>(meta_.d) * static_cast<std::size_t>(count));
    in.read(reinterpret_cast<char*>(out->data.data()),
            static_cast<std::streamsize>(out->data.size() * sizeof(float)));
    if (!in) {
        if (err) *err = "BaseListReader::ReadRawF32Span: read failed.";
        return false;
    }
    return true;
}

bool BuildBaseListStoreFromBasicBuckets(const std::string& base_basic_dir,
                                        const IvfListsReader& lists,
                                        const std::string& out_dir,
                                        const std::string& tmp_dir,
                                        bool keep_tmp,
                                        bool profile_timing,
                                        std::string* err) {
    (void)keep_tmp;  // temp files are no longer used (kept for API compatibility).
    if (base_basic_dir.empty() || out_dir.empty() || tmp_dir.empty()) {
        if (err) *err = "BuildBaseListStoreFromBasicBuckets: empty dir.";
        return false;
    }
    if (!EnsureDir(out_dir, err)) return false;
    if (!EnsureDir(tmp_dir, err)) return false;

    BaseBasicReader base_basic;
    if (!base_basic.Open(base_basic_dir, err)) {
        return false;
    }
    const BaseBasicMeta& bm = base_basic.meta();
    if (!bm.write_basic_to_bucket) {
        if (err) {
            *err = "BuildBaseListStoreFromBasicBuckets: base store does not have bucket basic payload "
                   "(write_basic_to_bucket=false).";
        }
        return false;
    }

    const int d = bm.d;
    const int m = bm.m;
    const int m_codes = bm.m_codes;
    const int raw_bytes_per_vec = bm.write_vector_bucket ? std::max(0, bm.raw_bytes_per_vec) : 0;
    const bool want_raw = (raw_bytes_per_vec > 0);
    const bool want_raw_u8 = want_raw && (raw_bytes_per_vec == d);
    const bool want_raw_f32 = want_raw && (raw_bytes_per_vec == d * static_cast<int>(sizeof(float)));
    if (want_raw && !want_raw_u8 && !want_raw_f32) {
        if (err) {
            *err = "BuildBaseListStoreFromBasicBuckets: unsupported raw_bytes_per_vec=" +
                   std::to_string(raw_bytes_per_vec) + " (d=" + std::to_string(d) + ").";
        }
        return false;
    }
    const int bucket_size = std::max(1, bm.bucket_size);
    const int nlist = lists.nlist();
    const int nbucket = (nlist + bucket_size - 1) / bucket_size;
    const std::uint64_t ntotal = lists.ntotal();

    // meta.bin
    {
        const std::string meta_path = JoinPath(out_dir, "meta.bin");
        std::ofstream meta(meta_path, std::ios::binary | std::ios::trunc);
        if (!meta.is_open()) {
            if (err) *err = "BuildBaseListStoreFromBasicBuckets: failed to open " + meta_path;
            return false;
        }
        struct Header {
            std::uint32_t magic = 0x534C5342u;  // "BSLS"
            std::uint32_t version = 1;
            std::int32_t d = 0;
            std::int32_t m = 0;
            std::int32_t m_codes = 0;
            std::uint32_t reserved = 0;
            std::uint64_t ntotal = 0;
        } h;
        h.d = d;
        h.m = m;
        h.m_codes = m_codes;
        h.ntotal = ntotal;
        meta.write(reinterpret_cast<const char*>(&h), sizeof(Header));
        if (!meta) {
            if (err) *err = "BuildBaseListStoreFromBasicBuckets: failed to write meta.bin.";
            return false;
        }
    }

    const std::string codes_out_path = JoinPath(out_dir, "codes.bin");
    const std::string coeffs_out_path = JoinPath(out_dir, "coeffs.bin");
    const std::string raw_u8_out_path = JoinPath(out_dir, "raw_u8.bin");
    const std::string raw_f32_out_path = JoinPath(out_dir, "raw_f32.bin");
    const std::string raw_out_path = want_raw_u8 ? raw_u8_out_path : raw_f32_out_path;

    const std::uint64_t codes_bytes_total = static_cast<std::uint64_t>(m_codes) * ntotal;
    const std::uint64_t coeffs_bytes_total = static_cast<std::uint64_t>(m) * ntotal * sizeof(float);
    const std::uint64_t raw_bytes_total =
        want_raw ? static_cast<std::uint64_t>(raw_bytes_per_vec) * ntotal : 0u;

    auto prealloc = [&](const std::string& path, std::uint64_t bytes) -> bool {
        {
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            if (!out.is_open()) {
                if (err) *err = "BuildBaseListStoreFromBasicBuckets: failed to create " + path;
                return false;
            }
        }
        try {
            std::filesystem::resize_file(path, static_cast<std::uintmax_t>(bytes));
        } catch (...) {
            if (err) *err = "BuildBaseListStoreFromBasicBuckets: failed to resize " + path;
            return false;
        }
        return true;
    };
    if (!prealloc(codes_out_path, codes_bytes_total)) return false;
    if (!prealloc(coeffs_out_path, coeffs_bytes_total)) return false;
    if (want_raw) {
        if (!prealloc(raw_out_path, raw_bytes_total)) return false;
        // Avoid stale raw file of the other type from previous runs.
        std::error_code ec;
        if (want_raw_u8) {
            std::filesystem::remove(raw_f32_out_path, ec);
        } else if (want_raw_f32) {
            std::filesystem::remove(raw_u8_out_path, ec);
        }
    } else {
        std::error_code ec;
        std::filesystem::remove(raw_u8_out_path, ec);
        std::filesystem::remove(raw_f32_out_path, ec);
    }

#if defined(_WIN32)
    std::fstream codes_rw(codes_out_path, std::ios::binary | std::ios::in | std::ios::out);
    std::fstream coeffs_rw(coeffs_out_path, std::ios::binary | std::ios::in | std::ios::out);
    std::fstream raw_rw;
    if (!codes_rw.is_open() || !coeffs_rw.is_open()) {
        if (err) *err = "BuildBaseListStoreFromBasicBuckets: failed to open payload files.";
        return false;
    }
    if (want_raw) {
        raw_rw.open(raw_out_path, std::ios::binary | std::ios::in | std::ios::out);
        if (!raw_rw.is_open()) {
            if (err) *err = "BuildBaseListStoreFromBasicBuckets: failed to open raw vectors file.";
            return false;
        }
    }
#else
    int fd_codes = OpenRw(codes_out_path, err);
    if (fd_codes < 0) return false;
    int fd_coeffs = OpenRw(coeffs_out_path, err);
    if (fd_coeffs < 0) {
        CloseFd(&fd_codes);
        return false;
    }
    int fd_raw = -1;
    if (want_raw) {
        fd_raw = OpenRw(raw_out_path, err);
        if (fd_raw < 0) {
            CloseFd(&fd_codes);
            CloseFd(&fd_coeffs);
            return false;
        }
    }
#endif

    // Fixed-size bucket record format (as written by BaseBasicWriter).
    const std::size_t rec_bytes =
        2u * sizeof(std::uint32_t) +
        (want_raw ? static_cast<std::size_t>(raw_bytes_per_vec) : 0u) +
        static_cast<std::size_t>(m_codes) +
        static_cast<std::size_t>(m) * sizeof(float);

    const std::size_t off_raw = 2u * sizeof(std::uint32_t);
    const std::size_t off_basic =
        2u * sizeof(std::uint32_t) + (want_raw ? static_cast<std::size_t>(raw_bytes_per_vec) : 0u);
    const std::size_t bytes_coeffs = static_cast<std::size_t>(m) * sizeof(float);

    std::vector<std::uint8_t> rec(rec_bytes);
    std::vector<std::uint64_t> next_pos(static_cast<std::size_t>(nlist), 0);
    for (int cid = 0; cid < nlist; ++cid) {
        next_pos[static_cast<std::size_t>(cid)] = lists.offsets()[static_cast<std::size_t>(cid)];
    }

    constexpr std::size_t kFlushBytes = 4u * 1024u * 1024u;
    auto now_s = []() -> double {
        using clock = std::chrono::steady_clock;
        return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
    };
    const double t_total0 = profile_timing ? now_s() : 0.0;
    double t_bucket_total = 0.0;
    double t_write_codes = 0.0;
    double t_write_coeffs = 0.0;
    double t_write_raw = 0.0;

    auto append_bytes = [](std::vector<std::uint8_t>* dst, const void* src, std::size_t bytes) {
        if (bytes == 0) return;
        const auto* p = static_cast<const std::uint8_t*>(src);
        // Avoid `resize()+memcpy` because resize() value-initializes (zero-fills) new bytes.
        dst->insert(dst->end(), p, p + bytes);
    };

    for (int b = 0; b < nbucket; ++b) {
        const int c0 = b * bucket_size;
        const int c1 = std::min(nlist, (b + 1) * bucket_size);
        const int nlocal = c1 - c0;

        const std::string bucket_tag = ZeroPad(b, 4);
        const std::string bucket_path = JoinPath(base_basic_dir, "bucket_" + bucket_tag + ".bin");
        std::ifstream in(bucket_path, std::ios::binary);
        if (!in.is_open()) {
            // Some buckets may be empty and not present. However, when the IVF lists indicate that
            // any cluster in this bucket range has non-empty list content, the bucket payload must exist.
            // Missing bucket files at this point usually means base_basic payloads were auto-cleaned,
            // and base_list cannot be rebuilt without re-running base.encode.
            bool needed = false;
            for (int cid = c0; cid < c1; ++cid) {
                if (lists.ListSize(cid) > 0) {
                    needed = true;
                    break;
                }
            }
            if (needed) {
                if (err) {
                    *err = "BuildBaseListStoreFromBasicBuckets: missing bucket payload file: " + bucket_path +
                           " ; base_basic buckets may have been auto-cleaned; enable base.encode to rebuild.";
                }
                return false;
            }
            continue;
        }

        const double t_bucket0 = profile_timing ? now_s() : 0.0;
        std::vector<std::vector<std::uint8_t>> codes_buf(static_cast<std::size_t>(nlocal));
        std::vector<std::vector<std::uint8_t>> coeffs_buf(static_cast<std::size_t>(nlocal));
        std::vector<std::vector<std::uint8_t>> raw_buf;
        if (want_raw) {
            raw_buf.resize(static_cast<std::size_t>(nlocal));
        }
        std::vector<std::uint64_t> buf_begin_pos(static_cast<std::size_t>(nlocal), 0);
        std::vector<std::uint32_t> buf_count(static_cast<std::size_t>(nlocal), 0);
        std::vector<std::uint8_t> buf_active(static_cast<std::size_t>(nlocal), 0);

        auto flush_cluster = [&](int lc) -> bool {
            const auto idx = static_cast<std::size_t>(lc);
            if (!buf_active[idx] || buf_count[idx] == 0) {
                buf_active[idx] = 0;
                buf_count[idx] = 0;
                codes_buf[idx].clear();
                coeffs_buf[idx].clear();
                if (want_raw) raw_buf[idx].clear();
                return true;
            }

            const std::uint64_t pos0 = buf_begin_pos[idx];
            if (m_codes > 0 && !codes_buf[idx].empty()) {
                const std::uint64_t off = pos0 * static_cast<std::uint64_t>(m_codes);
                const double t0 = profile_timing ? now_s() : 0.0;
#if defined(_WIN32)
                if (!SeekWriteAll(&codes_rw, off, codes_buf[idx].data(), codes_buf[idx].size(), err)) return false;
#else
                if (!PwriteAll(fd_codes, codes_buf[idx].data(), codes_buf[idx].size(), off, err)) return false;
#endif
                if (profile_timing) {
                    t_write_codes += now_s() - t0;
                }
            }
            if (!coeffs_buf[idx].empty()) {
                const std::uint64_t off = pos0 * static_cast<std::uint64_t>(m) * sizeof(float);
                const double t0 = profile_timing ? now_s() : 0.0;
#if defined(_WIN32)
                if (!SeekWriteAll(&coeffs_rw, off, coeffs_buf[idx].data(), coeffs_buf[idx].size(), err)) return false;
#else
                if (!PwriteAll(fd_coeffs, coeffs_buf[idx].data(), coeffs_buf[idx].size(), off, err)) return false;
#endif
                if (profile_timing) {
                    t_write_coeffs += now_s() - t0;
                }
            }
            if (want_raw && !raw_buf[idx].empty()) {
                const std::uint64_t off = pos0 * static_cast<std::uint64_t>(raw_bytes_per_vec);
                const double t0 = profile_timing ? now_s() : 0.0;
#if defined(_WIN32)
                if (!SeekWriteAll(&raw_rw, off, raw_buf[idx].data(), raw_buf[idx].size(), err)) return false;
#else
                if (!PwriteAll(fd_raw, raw_buf[idx].data(), raw_buf[idx].size(), off, err)) return false;
#endif
                if (profile_timing) {
                    t_write_raw += now_s() - t0;
                }
            }

            buf_active[idx] = 0;
            buf_count[idx] = 0;
            codes_buf[idx].clear();
            coeffs_buf[idx].clear();
            if (want_raw) raw_buf[idx].clear();
            return true;
        };

        while (ReadAll(in, rec.data(), rec_bytes)) {
            std::uint32_t cid = 0;
            std::memcpy(&cid, rec.data() + sizeof(std::uint32_t), sizeof(std::uint32_t));
            const int lc = static_cast<int>(cid) - c0;
            if (lc < 0 || lc >= nlocal) {
                if (err) *err = "BuildBaseListStoreFromBasicBuckets: bucket record has cid outside bucket range.";
                return false;
            }
            const auto idx = static_cast<std::size_t>(lc);
            if (!buf_active[idx]) {
                buf_begin_pos[idx] = next_pos[static_cast<std::size_t>(cid)];
                buf_active[idx] = 1;
                buf_count[idx] = 0;
            }

            if (want_raw) {
                append_bytes(&raw_buf[idx], rec.data() + off_raw, static_cast<std::size_t>(raw_bytes_per_vec));
            }
            append_bytes(&codes_buf[idx], rec.data() + off_basic, static_cast<std::size_t>(m_codes));
            append_bytes(&coeffs_buf[idx], rec.data() + off_basic + static_cast<std::size_t>(m_codes), bytes_coeffs);
            ++buf_count[idx];
            ++next_pos[static_cast<std::size_t>(cid)];

            const std::size_t total_bytes =
                codes_buf[idx].size() + coeffs_buf[idx].size() + (want_raw ? raw_buf[idx].size() : 0u);
            if (total_bytes >= kFlushBytes) {
                if (!flush_cluster(lc)) return false;
            }
        }

        for (int lc = 0; lc < nlocal; ++lc) {
            if (!flush_cluster(lc)) return false;
        }

        if (profile_timing) {
            t_bucket_total += now_s() - t_bucket0;
        }
    }

#if !defined(_WIN32)
    CloseFd(&fd_codes);
    CloseFd(&fd_coeffs);
    if (want_raw) CloseFd(&fd_raw);
#endif

    // Validate that we filled every slot indicated by IVF offsets (catch silent corruption).
    for (int cid = 0; cid < nlist; ++cid) {
        const std::uint64_t begin = lists.offsets()[static_cast<std::size_t>(cid)];
        const std::uint64_t expect = begin + static_cast<std::uint64_t>(lists.ListSize(cid));
        const std::uint64_t got = next_pos[static_cast<std::size_t>(cid)];
        if (got != expect) {
            if (err) {
                *err = "BuildBaseListStoreFromBasicBuckets: incomplete base_list rebuild (cid=" +
                       std::to_string(cid) + " expected_pos=" + std::to_string(expect) +
                       " got_pos=" + std::to_string(got) + ").";
            }
            return false;
        }
    }

    if (profile_timing) {
        const double total_s = now_s() - t_total0;
        LogInfo("Base list-store from buckets timing(s): total=" +
                FormatFloat(static_cast<float>(total_s), 3) +
                " bucket_io=" + FormatFloat(static_cast<float>(t_bucket_total), 3) +
                " write_codes=" + FormatFloat(static_cast<float>(t_write_codes), 3) +
                " write_coeffs=" + FormatFloat(static_cast<float>(t_write_coeffs), 3) +
                " write_raw=" + FormatFloat(static_cast<float>(t_write_raw), 3));
    }

    return true;
}

BaseListThreadReader::~BaseListThreadReader() {
    Close();
}

void BaseListThreadReader::Close() {
    if (codes_in_.is_open()) codes_in_.close();
    if (coeffs_in_.is_open()) coeffs_in_.close();
    if (raw_u8_in_.is_open()) raw_u8_in_.close();
    if (raw_f32_in_.is_open()) raw_f32_in_.close();
    dir_.clear();
    meta_ = {};
}

bool BaseListThreadReader::OpenFiles(std::string* err) {
    if (dir_.empty() || meta_.m <= 0 || meta_.m_codes < 0) {
        if (err) *err = "BaseListThreadReader::Open: invalid meta/dir.";
        return false;
    }
    codes_in_.open(JoinPath(dir_, "codes.bin"), std::ios::binary);
    coeffs_in_.open(JoinPath(dir_, "coeffs.bin"), std::ios::binary);
    if (!codes_in_.is_open()) {
        if (err) *err = "BaseListThreadReader::Open: failed to open codes.bin.";
        return false;
    }
    if (meta_.has_raw_u8) {
        raw_u8_in_.open(JoinPath(dir_, "raw_u8.bin"), std::ios::binary);
        if (!raw_u8_in_.is_open()) {
            if (err) *err = "BaseListThreadReader::Open: failed to open raw_u8.bin.";
            return false;
        }
    }
    if (meta_.has_raw_f32) {
        raw_f32_in_.open(JoinPath(dir_, "raw_f32.bin"), std::ios::binary);
        if (!raw_f32_in_.is_open()) {
            if (err) *err = "BaseListThreadReader::Open: failed to open raw_f32.bin.";
            return false;
        }
    }
    return true;
}

bool BaseListThreadReader::OpenFrom(const BaseListReader& src, std::string* err) {
    Close();
    dir_ = src.dir();
    meta_ = src.meta();
    return OpenFiles(err);
}

bool BaseListThreadReader::Open(const std::string& dir, std::string* err) {
    BaseListReader tmp;
    if (!tmp.Open(dir, err)) {
        return false;
    }
    return OpenFrom(tmp, err);
}

static bool ReadAt(std::ifstream* in,
                   std::uint64_t off,
                   void* dst,
                   std::size_t bytes,
                   std::string* err) {
    if (bytes == 0) {
        return true;
    }
    in->seekg(static_cast<std::streamoff>(off), std::ios::beg);
    if (!(*in)) {
        if (err) *err = "BaseListThreadReader: seek failed.";
        return false;
    }
    in->read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes));
    if (!(*in)) {
        if (err) *err = "BaseListThreadReader: read failed.";
        return false;
    }
    return true;
}

bool BaseListThreadReader::ReadCodesSpan(std::uint64_t begin,
                                        std::uint32_t count,
                                        std::vector<Code>* out,
                                        std::string* err) {
    if (!out) {
        if (err) *err = "BaseListThreadReader::ReadCodesSpan: null output.";
        return false;
    }
    out->clear();
    if (count == 0 || meta_.m_codes == 0) {
        return true;
    }
    const auto rec = static_cast<std::uint64_t>(meta_.m_codes);
    const std::uint64_t off = begin * rec;
    const std::size_t bytes = static_cast<std::size_t>(rec) * static_cast<std::size_t>(count);
    out->resize(bytes);
    return ReadAt(&codes_in_, off, out->data(), bytes, err);
}

bool BaseListThreadReader::ReadCoeffsSpan(std::uint64_t begin,
                                         std::uint32_t count,
                                         std::vector<float>* out,
                                         std::string* err) {
    if (!out) {
        if (err) *err = "BaseListThreadReader::ReadCoeffsSpan: null output.";
        return false;
    }
    out->clear();
    if (count == 0) {
        return true;
    }
    if (!coeffs_in_.is_open()) {
        if (err) *err = "BaseListThreadReader::ReadCoeffsSpan: coeffs.bin is not available.";
        return false;
    }
    const std::uint64_t rec = static_cast<std::uint64_t>(meta_.m) * sizeof(float);
    const std::uint64_t off = begin * rec;
    const std::size_t n = static_cast<std::size_t>(meta_.m) * static_cast<std::size_t>(count);
    out->resize(n);
    return ReadAt(&coeffs_in_, off, out->data(), n * sizeof(float), err);
}

bool BaseListThreadReader::ReadRawU8Span(std::uint64_t begin,
                                        std::uint32_t count,
                                        ColMajorMatrix<std::uint8_t>* out,
                                        std::string* err) {
    if (!out) {
        if (err) *err = "BaseListThreadReader::ReadRawU8Span: null output.";
        return false;
    }
    out->rows = 0;
    out->cols = 0;
    out->data.clear();
    if (!meta_.has_raw_u8) {
        if (err) *err = "BaseListThreadReader::ReadRawU8Span: raw_u8.bin is not available.";
        return false;
    }
    if (count == 0) {
        return true;
    }
    const auto rec = static_cast<std::uint64_t>(meta_.raw_u8_bytes_per_vec);
    const std::uint64_t off = begin * rec;
    const std::size_t bytes = static_cast<std::size_t>(rec) * static_cast<std::size_t>(count);
    out->rows = meta_.d;
    out->cols = static_cast<int>(count);
    out->data.resize(bytes);
    return ReadAt(&raw_u8_in_, off, out->data.data(), bytes, err);
}

bool BaseListThreadReader::ReadRawF32Span(std::uint64_t begin,
                                         std::uint32_t count,
                                         ColMajorMatrix<float>* out,
                                         std::string* err) {
    if (!out) {
        if (err) *err = "BaseListThreadReader::ReadRawF32Span: null output.";
        return false;
    }
    out->rows = 0;
    out->cols = 0;
    out->data.clear();
    if (!meta_.has_raw_f32) {
        if (err) *err = "BaseListThreadReader::ReadRawF32Span: raw_f32.bin is not available.";
        return false;
    }
    if (count == 0) {
        return true;
    }
    const std::uint64_t rec = static_cast<std::uint64_t>(meta_.d) * sizeof(float);
    const std::uint64_t off = begin * rec;
    const std::size_t n = static_cast<std::size_t>(meta_.d) * static_cast<std::size_t>(count);
    out->rows = meta_.d;
    out->cols = static_cast<int>(count);
    out->data.resize(n);
    return ReadAt(&raw_f32_in_, off, out->data.data(), n * sizeof(float), err);
}

}  // namespace stlq::io
