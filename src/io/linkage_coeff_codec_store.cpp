#include "stlq/io/linkage_coeff_codec_store.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <system_error>

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
        if (err) *err = "LinkageCoeffCodecStore: failed to create dir: " + dir;
        return false;
    }
}

int StreamsFromGranularity(const std::string& g, int m) {
    std::string gg = g;
    std::transform(gg.begin(), gg.end(), gg.begin(), [](unsigned char c) { return std::tolower(c); });
    if (gg == "cluster" || gg == "per_cluster" || gg == "all" || gg == "all_layers") {
        return 1;
    }
    if (gg == "layer" || gg == "per_layer") {
        return std::max(1, m);
    }
    return 0;
}

int AlphabetSizeFromCfg(int max_bits, int alphabet_size) {
    if (alphabet_size == 256) {
        return 256;
    }
    const int Qmax = (1 << std::max(1, max_bits - 1)) - 1;
    return 2 * Qmax + 1;
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
                *err = "LinkageCoeffCodecStore: pwrite failed: " + std::string(std::strerror(errno));
            }
            return false;
        }
        if (w == 0) {
            if (err) *err = "LinkageCoeffCodecStore: pwrite returned 0.";
            return false;
        }
        remaining -= static_cast<std::size_t>(w);
        p += w;
        cur += static_cast<std::uint64_t>(w);
    }
    return true;
}

int OpenRwTrunc(const std::string& path, std::string* err) {
    const int fd = ::open(path.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0644);
    if (fd < 0) {
        if (err) {
            *err = "LinkageCoeffCodecStore: open failed: " + path + " : " + std::string(std::strerror(errno));
        }
        return -1;
    }
    return fd;
}

int OpenRwNoTrunc(const std::string& path, std::string* err) {
    const int fd = ::open(path.c_str(), O_RDWR, 0644);
    if (fd < 0) {
        if (err) {
            *err = "LinkageCoeffCodecStore: open(no-trunc) failed: " + path + " : " +
                   std::string(std::strerror(errno));
        }
        return -1;
    }
    return fd;
}

#endif  // !_WIN32

template <typename T>
bool WriteBinaryFile(const std::string& path, const std::vector<T>& v, std::string* err) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        if (err) *err = "LinkageCoeffCodecStore: failed to write: " + path;
        return false;
    }
    if (!v.empty()) {
        out.write(reinterpret_cast<const char*>(v.data()),
                  static_cast<std::streamsize>(v.size() * sizeof(T)));
        if (!out) {
            if (err) *err = "LinkageCoeffCodecStore: write failed: " + path;
            return false;
        }
    }
    return true;
}

template <typename T>
bool ReadBinaryFile(const std::string& path, std::vector<T>* v, std::string* err) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        if (err) *err = "LinkageCoeffCodecStore: failed to open: " + path;
        return false;
    }
    in.seekg(0, std::ios::end);
    const std::streamoff bytes = in.tellg();
    if (bytes < 0) {
        if (err) *err = "LinkageCoeffCodecStore: tellg failed: " + path;
        return false;
    }
    if (bytes % static_cast<std::streamoff>(sizeof(T)) != 0) {
        if (err) *err = "LinkageCoeffCodecStore: size mismatch: " + path;
        return false;
    }
    const auto n = static_cast<std::size_t>(bytes / static_cast<std::streamoff>(sizeof(T)));
    v->resize(n);
    in.seekg(0, std::ios::beg);
    if (n > 0) {
        in.read(reinterpret_cast<char*>(v->data()), static_cast<std::streamsize>(n * sizeof(T)));
        if (!in) {
            if (err) *err = "LinkageCoeffCodecStore: read failed: " + path;
            return false;
        }
    }
    return true;
}

}  // namespace

LinkageCoeffCodecThreadReader::~LinkageCoeffCodecThreadReader() { Close(); }

void LinkageCoeffCodecThreadReader::Close() {
    scales_in_.close();
    lens_in_.close();
    payload_in_.close();
    src_ = nullptr;
    dir_.clear();
    meta_ = {};
}

bool LinkageCoeffCodecThreadReader::ReadAt(std::ifstream* in,
                                        std::uint64_t off,
                                        void* dst,
                                        std::size_t bytes,
                                        std::string* err) {
    if (!in || !in->is_open()) {
        if (err) *err = "LinkageCoeffCodecThreadReader: file not open.";
        return false;
    }
    in->seekg(static_cast<std::streamoff>(off), std::ios::beg);
    if (!(*in)) {
        if (err) *err = "LinkageCoeffCodecThreadReader: seekg failed.";
        return false;
    }
    if (bytes > 0) {
        in->read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes));
        if (!(*in)) {
            if (err) *err = "LinkageCoeffCodecThreadReader: read failed.";
            return false;
        }
    }
    return true;
}

bool LinkageCoeffCodecThreadReader::OpenFiles(std::string* err) {
    scales_in_.open(JoinPath(dir_, "coeff_scales.f32"), std::ios::binary);
    lens_in_.open(JoinPath(dir_, "coeff_lens.u8"), std::ios::binary);
    payload_in_.open(JoinPath(dir_, "coeff_payload.bin"), std::ios::binary);
    if (!scales_in_.is_open()) {
        if (err) *err = "LinkageCoeffCodecThreadReader: missing scales file.";
        return false;
    }
    if (!lens_in_.is_open()) {
        if (err) *err = "LinkageCoeffCodecThreadReader: missing lens file.";
        return false;
    }
    if (!payload_in_.is_open()) {
        if (err) *err = "LinkageCoeffCodecThreadReader: missing payload file.";
        return false;
    }
    return true;
}

bool LinkageCoeffCodecThreadReader::OpenFrom(const LinkageCoeffCodecReader& src, std::string* err) {
    Close();
    src_ = &src;
    dir_ = src.dir();
    meta_ = src.meta();
    if (dir_.empty()) {
        if (err) *err = "LinkageCoeffCodecThreadReader: empty dir.";
        return false;
    }
    if (meta_.nlist <= 0 || meta_.m <= 0 || meta_.streams <= 0 || meta_.S <= 0) {
        if (err) *err = "LinkageCoeffCodecThreadReader: invalid meta.";
        return false;
    }
    if (!OpenFiles(err)) {
        Close();
        return false;
    }
    return true;
}

bool LinkageCoeffCodecThreadReader::ReadCluster(int cid,
                                             std::vector<float>* scales_root,
                                             std::vector<float>* scales_linkage,
                                             std::vector<std::vector<std::uint8_t>>* lens_root,
                                             std::vector<std::vector<std::uint8_t>>* lens_linkage,
                                             std::vector<std::vector<std::uint8_t>>* payload_root,
                                             std::vector<std::vector<std::uint8_t>>* payload_linkage,
                                             std::string* err) {
    if (!src_) {
        if (err) *err = "LinkageCoeffCodecThreadReader: not opened.";
        return false;
    }
    if (cid < 0 || cid >= meta_.nlist) {
        if (err) *err = "LinkageCoeffCodecThreadReader: cid out of range.";
        return false;
    }
    if (!scales_root || !scales_linkage || !lens_root || !lens_linkage || !payload_root || !payload_linkage) {
        if (err) *err = "LinkageCoeffCodecThreadReader: null output.";
        return false;
    }

    const int streams = meta_.streams;
    const int S = meta_.S;

    scales_root->assign(static_cast<std::size_t>(meta_.m), 1.0f);
    scales_linkage->assign(static_cast<std::size_t>(meta_.m), 1.0f);
    lens_root->assign(static_cast<std::size_t>(streams), std::vector<std::uint8_t>(static_cast<std::size_t>(S), 0));
    lens_linkage->assign(static_cast<std::size_t>(streams), std::vector<std::uint8_t>(static_cast<std::size_t>(S), 0));
    payload_root->assign(static_cast<std::size_t>(streams), {});
    payload_linkage->assign(static_cast<std::size_t>(streams), {});

    // Read scales (fixed-size).
    {
        std::vector<float> tmp(static_cast<std::size_t>(2 * meta_.m));
        const std::uint64_t off = static_cast<std::uint64_t>(cid) * 2ull *
                                  static_cast<std::uint64_t>(meta_.m) * sizeof(float);
        if (!ReadAt(&scales_in_, off, tmp.data(), tmp.size() * sizeof(float), err)) {
            return false;
        }
        std::copy(tmp.begin(), tmp.begin() + meta_.m, scales_root->begin());
        std::copy(tmp.begin() + meta_.m, tmp.end(), scales_linkage->begin());
    }

    auto load_group = [&](int group,
                          std::vector<std::vector<std::uint8_t>>* lens_out,
                          std::vector<std::vector<std::uint8_t>>* payload_out) -> bool {
        const auto& offs_u64 = src_->payload_offs_u64();
        const auto& sizes_u32 = src_->payload_sizes_u32();
        for (int s = 0; s < streams; ++s) {
            const std::size_t idx =
                (static_cast<std::size_t>(group) * static_cast<std::size_t>(meta_.nlist) +
                 static_cast<std::size_t>(cid)) *
                    static_cast<std::size_t>(streams) +
                static_cast<std::size_t>(s);

            const std::uint64_t off_lens = static_cast<std::uint64_t>(idx) *
                                           static_cast<std::uint64_t>(S);
            if (!ReadAt(&lens_in_, off_lens,
                        lens_out->at(static_cast<std::size_t>(s)).data(),
                        static_cast<std::size_t>(S),
                        err)) {
                return false;
            }

            const std::uint64_t off_payload = offs_u64[idx];
            const std::uint32_t sz = sizes_u32[idx];
            payload_out->at(static_cast<std::size_t>(s)).resize(sz);
            if (sz > 0) {
                if (!ReadAt(&payload_in_, off_payload,
                            payload_out->at(static_cast<std::size_t>(s)).data(),
                            static_cast<std::size_t>(sz),
                            err)) {
                    return false;
                }
            }
        }
        return true;
    };

    if (!load_group(0, lens_root, payload_root)) return false;
    if (!load_group(1, lens_linkage, payload_linkage)) return false;
    return true;
}

bool LinkageCoeffCodecRandomWriter::Open(const LinkageCoeffCodecStoreConfig& cfg, std::string* err) {
    cfg_ = cfg;
    if (cfg_.dir.empty()) {
        if (err) *err = "LinkageCoeffCodecStore: empty dir.";
        return false;
    }
    if (cfg_.nlist <= 0 || cfg_.m <= 0) {
        if (err) *err = "LinkageCoeffCodecStore: invalid nlist/m.";
        return false;
    }

    meta_.nlist = cfg_.nlist;
    meta_.m = cfg_.m;
    meta_.max_bits = std::max(2, cfg_.max_bits);
    meta_.alphabet_size = cfg_.alphabet_size;
    meta_.streams = StreamsFromGranularity(cfg_.granularity, cfg_.m);
    if (meta_.streams <= 0) {
        if (err) *err = "LinkageCoeffCodecStore: invalid granularity: " + cfg_.granularity;
        return false;
    }
    meta_.S = AlphabetSizeFromCfg(meta_.max_bits, meta_.alphabet_size);
    if (meta_.S <= 0) {
        if (err) *err = "LinkageCoeffCodecStore: invalid alphabet size.";
        return false;
    }

    if (!EnsureDir(cfg_.dir, err)) {
        return false;
    }

    const std::size_t nstreams_all = static_cast<std::size_t>(meta_.nlist) * 2u *
                                     static_cast<std::size_t>(meta_.streams);
    payload_offs_u64_.assign(nstreams_all, 0);
    payload_sizes_u32_.assign(nstreams_all, 0);

    const std::uint64_t lens_bytes =
        static_cast<std::uint64_t>(nstreams_all) * static_cast<std::uint64_t>(meta_.S);
    const std::uint64_t scales_bytes =
        static_cast<std::uint64_t>(meta_.nlist) * 2ull *
        static_cast<std::uint64_t>(meta_.m) * sizeof(float);

    stats_.lens_bytes = lens_bytes;
    stats_.scales_bytes = scales_bytes;
    stats_.payload_offsets_bytes = static_cast<std::uint64_t>(nstreams_all) * sizeof(std::uint64_t);
    stats_.payload_sizes_bytes = static_cast<std::uint64_t>(nstreams_all) * sizeof(std::uint32_t);

    if (!WriteMeta(err)) {
        return false;
    }

    const std::string payload_path = JoinPath(cfg_.dir, "coeff_payload.bin");
    const std::string lens_path = JoinPath(cfg_.dir, "coeff_lens.u8");
    const std::string scales_path = JoinPath(cfg_.dir, "coeff_scales.f32");
    const std::string offs_path = JoinPath(cfg_.dir, "coeff_payload_offs.u64");
    const std::string sizes_path = JoinPath(cfg_.dir, "coeff_payload_sizes.u32");

#if defined(_WIN32)
    payload_out_.open(payload_path, std::ios::binary | std::ios::trunc | std::ios::in | std::ios::out);
    lens_out_.open(lens_path, std::ios::binary | std::ios::trunc | std::ios::in | std::ios::out);
    scales_out_.open(scales_path, std::ios::binary | std::ios::trunc | std::ios::in | std::ios::out);
    payload_offs_out_.open(offs_path, std::ios::binary | std::ios::trunc | std::ios::in | std::ios::out);
    payload_sizes_out_.open(sizes_path, std::ios::binary | std::ios::trunc | std::ios::in | std::ios::out);
    if (!payload_out_.is_open() || !lens_out_.is_open() || !scales_out_.is_open() ||
        !payload_offs_out_.is_open() || !payload_sizes_out_.is_open()) {
        if (err) *err = "LinkageCoeffCodecStore: failed to open codec files.";
        return false;
    }
    // Pre-extend fixed-size files so random writes do not hit EOF failures.
    if (lens_bytes > 0) {
        lens_out_.seekp(static_cast<std::streamoff>(lens_bytes - 1), std::ios::beg);
        lens_out_.write("", 1);
    }
    if (scales_bytes > 0) {
        scales_out_.seekp(static_cast<std::streamoff>(scales_bytes - 1), std::ios::beg);
        scales_out_.write("", 1);
    }
    if (stats_.payload_offsets_bytes > 0) {
        payload_offs_out_.seekp(static_cast<std::streamoff>(stats_.payload_offsets_bytes - 1), std::ios::beg);
        payload_offs_out_.write("", 1);
    }
    if (stats_.payload_sizes_bytes > 0) {
        payload_sizes_out_.seekp(static_cast<std::streamoff>(stats_.payload_sizes_bytes - 1), std::ios::beg);
        payload_sizes_out_.write("", 1);
    }
#else
    fd_payload_ = OpenRwTrunc(payload_path, err);
    fd_lens_ = OpenRwTrunc(lens_path, err);
    fd_scales_ = OpenRwTrunc(scales_path, err);
    fd_payload_offs_ = OpenRwTrunc(offs_path, err);
    fd_payload_sizes_ = OpenRwTrunc(sizes_path, err);
    if (fd_payload_ < 0 || fd_lens_ < 0 || fd_scales_ < 0 || fd_payload_offs_ < 0 || fd_payload_sizes_ < 0) {
        return false;
    }
    if (::ftruncate(fd_lens_, static_cast<off_t>(lens_bytes)) != 0) {
        if (err) *err = "LinkageCoeffCodecStore: ftruncate lens failed.";
        return false;
    }
    if (::ftruncate(fd_scales_, static_cast<off_t>(scales_bytes)) != 0) {
        if (err) *err = "LinkageCoeffCodecStore: ftruncate scales failed.";
        return false;
    }
    if (::ftruncate(fd_payload_offs_, static_cast<off_t>(stats_.payload_offsets_bytes)) != 0) {
        if (err) *err = "LinkageCoeffCodecStore: ftruncate payload_offs failed.";
        return false;
    }
    if (::ftruncate(fd_payload_sizes_, static_cast<off_t>(stats_.payload_sizes_bytes)) != 0) {
        if (err) *err = "LinkageCoeffCodecStore: ftruncate payload_sizes failed.";
        return false;
    }
#endif

    return true;
}

bool LinkageCoeffCodecRandomWriter::OpenResume(const LinkageCoeffCodecStoreConfig& cfg, std::string* err) {
    cfg_ = cfg;
    if (cfg_.dir.empty()) {
        if (err) *err = "LinkageCoeffCodecStore: OpenResume empty dir.";
        return false;
    }
    if (cfg_.nlist <= 0 || cfg_.m <= 0) {
        if (err) *err = "LinkageCoeffCodecStore: OpenResume invalid nlist/m.";
        return false;
    }
    if (!EnsureDir(cfg_.dir, err)) {
        return false;
    }

    // Read meta written by the initial run.
    {
        const std::string meta_path = JoinPath(cfg_.dir, "coeff_meta.bin");
        std::ifstream in(meta_path, std::ios::binary);
        if (!in.is_open()) {
            if (err) *err = "LinkageCoeffCodecStore: OpenResume missing meta: " + meta_path;
            return false;
        }
        std::uint32_t magic = 0;
        in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        if (magic != 0x51434343u) {
            if (err) *err = "LinkageCoeffCodecStore: OpenResume bad meta magic.";
            return false;
        }
        in.read(reinterpret_cast<char*>(&meta_.version), sizeof(meta_.version));
        in.read(reinterpret_cast<char*>(&meta_.nlist), sizeof(meta_.nlist));
        in.read(reinterpret_cast<char*>(&meta_.m), sizeof(meta_.m));
        in.read(reinterpret_cast<char*>(&meta_.streams), sizeof(meta_.streams));
        in.read(reinterpret_cast<char*>(&meta_.max_bits), sizeof(meta_.max_bits));
        in.read(reinterpret_cast<char*>(&meta_.alphabet_size), sizeof(meta_.alphabet_size));
        in.read(reinterpret_cast<char*>(&meta_.S), sizeof(meta_.S));
        if (!in) {
            if (err) *err = "LinkageCoeffCodecStore: OpenResume meta read failed.";
            return false;
        }
    }
    if (meta_.nlist != cfg_.nlist || meta_.m != cfg_.m || meta_.streams <= 0 || meta_.S <= 0) {
        if (err) *err = "LinkageCoeffCodecStore: OpenResume meta/config mismatch.";
        return false;
    }

    const std::size_t nstreams_all = static_cast<std::size_t>(meta_.nlist) * 2u *
                                     static_cast<std::size_t>(meta_.streams);
    payload_offs_u64_.assign(nstreams_all, 0);
    payload_sizes_u32_.assign(nstreams_all, 0);

    const std::uint64_t lens_bytes =
        static_cast<std::uint64_t>(nstreams_all) * static_cast<std::uint64_t>(meta_.S);
    const std::uint64_t scales_bytes =
        static_cast<std::uint64_t>(meta_.nlist) * 2ull *
        static_cast<std::uint64_t>(meta_.m) * sizeof(float);

    stats_.lens_bytes = lens_bytes;
    stats_.scales_bytes = scales_bytes;
    stats_.payload_offsets_bytes = static_cast<std::uint64_t>(nstreams_all) * sizeof(std::uint64_t);
    stats_.payload_sizes_bytes = static_cast<std::uint64_t>(nstreams_all) * sizeof(std::uint32_t);

    const std::string payload_path = JoinPath(cfg_.dir, "coeff_payload.bin");
    const std::string lens_path = JoinPath(cfg_.dir, "coeff_lens.u8");
    const std::string scales_path = JoinPath(cfg_.dir, "coeff_scales.f32");
    const std::string offs_path = JoinPath(cfg_.dir, "coeff_payload_offs.u64");
    const std::string sizes_path = JoinPath(cfg_.dir, "coeff_payload_sizes.u32");

    auto check_size = [&](const std::string& path, std::uint64_t want) -> bool {
        std::error_code ec;
        if (!std::filesystem::exists(path, ec)) {
            if (err) *err = "LinkageCoeffCodecStore: OpenResume missing file: " + path;
            return false;
        }
        const std::uint64_t got = static_cast<std::uint64_t>(std::filesystem::file_size(path, ec));
        if (ec) {
            if (err) *err = "LinkageCoeffCodecStore: OpenResume stat failed: " + path;
            return false;
        }
        if (got != want) {
            if (err) *err = "LinkageCoeffCodecStore: OpenResume size mismatch: " + path +
                            " got=" + std::to_string(got) + " want=" + std::to_string(want);
            return false;
        }
        return true;
    };
    if (!check_size(lens_path, lens_bytes)) return false;
    if (!check_size(scales_path, scales_bytes)) return false;
    if (!check_size(offs_path, stats_.payload_offsets_bytes)) return false;
    if (!check_size(sizes_path, stats_.payload_sizes_bytes)) return false;
    {
        std::error_code ec;
        if (!std::filesystem::exists(payload_path, ec)) {
            if (err) *err = "LinkageCoeffCodecStore: OpenResume missing payload: " + payload_path;
            return false;
        }
    }

#if defined(_WIN32)
    payload_out_.open(payload_path, std::ios::binary | std::ios::in | std::ios::out);
    lens_out_.open(lens_path, std::ios::binary | std::ios::in | std::ios::out);
    scales_out_.open(scales_path, std::ios::binary | std::ios::in | std::ios::out);
    payload_offs_out_.open(offs_path, std::ios::binary | std::ios::in | std::ios::out);
    payload_sizes_out_.open(sizes_path, std::ios::binary | std::ios::in | std::ios::out);
    if (!payload_out_.is_open() || !lens_out_.is_open() || !scales_out_.is_open() ||
        !payload_offs_out_.is_open() || !payload_sizes_out_.is_open()) {
        if (err) *err = "LinkageCoeffCodecStore: OpenResume failed to open codec files.";
        return false;
    }
    {
        std::error_code ec;
        const std::uint64_t payload_bytes = static_cast<std::uint64_t>(std::filesystem::file_size(payload_path, ec));
        payload_cursor_.store(payload_bytes, std::memory_order_relaxed);
    }
#else
    fd_payload_ = OpenRwNoTrunc(payload_path, err);
    fd_lens_ = OpenRwNoTrunc(lens_path, err);
    fd_scales_ = OpenRwNoTrunc(scales_path, err);
    fd_payload_offs_ = OpenRwNoTrunc(offs_path, err);
    fd_payload_sizes_ = OpenRwNoTrunc(sizes_path, err);
    if (fd_payload_ < 0 || fd_lens_ < 0 || fd_scales_ < 0 || fd_payload_offs_ < 0 || fd_payload_sizes_ < 0) {
        return false;
    }
    {
        std::error_code ec;
        const std::uint64_t payload_bytes = static_cast<std::uint64_t>(std::filesystem::file_size(payload_path, ec));
        payload_cursor_.store(payload_bytes, std::memory_order_relaxed);
    }
#endif

    // Load existing offs/sizes arrays (best-effort; used for reader compatibility and potential debug).
    {
        std::ifstream in_offs(offs_path, std::ios::binary);
        std::ifstream in_sizes(sizes_path, std::ios::binary);
        if (in_offs.is_open()) {
            in_offs.read(reinterpret_cast<char*>(payload_offs_u64_.data()),
                         static_cast<std::streamsize>(payload_offs_u64_.size() * sizeof(std::uint64_t)));
        }
        if (in_sizes.is_open()) {
            in_sizes.read(reinterpret_cast<char*>(payload_sizes_u32_.data()),
                          static_cast<std::streamsize>(payload_sizes_u32_.size() * sizeof(std::uint32_t)));
        }
    }

    stats_.payload_bytes = payload_cursor_.load(std::memory_order_relaxed);
    return true;
}

bool LinkageCoeffCodecRandomWriter::WriteMeta(std::string* err) {
    const std::string meta_path = JoinPath(cfg_.dir, "coeff_meta.bin");
    std::ofstream out(meta_path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        if (err) *err = "LinkageCoeffCodecStore: failed to write meta: " + meta_path;
        return false;
    }
    const std::uint32_t magic = 0x51434343u;  // 'QCCC'
    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&meta_.version), sizeof(meta_.version));
    out.write(reinterpret_cast<const char*>(&meta_.nlist), sizeof(meta_.nlist));
    out.write(reinterpret_cast<const char*>(&meta_.m), sizeof(meta_.m));
    out.write(reinterpret_cast<const char*>(&meta_.streams), sizeof(meta_.streams));
    out.write(reinterpret_cast<const char*>(&meta_.max_bits), sizeof(meta_.max_bits));
    out.write(reinterpret_cast<const char*>(&meta_.alphabet_size), sizeof(meta_.alphabet_size));
    out.write(reinterpret_cast<const char*>(&meta_.S), sizeof(meta_.S));
    if (!out) {
        if (err) *err = "LinkageCoeffCodecStore: meta write failed.";
        return false;
    }
    return true;
}

bool LinkageCoeffCodecRandomWriter::WriteClusterAt(int cid,
                                                const std::vector<float>& scales_root,
                                                const std::vector<float>& scales_linkage,
                                                const std::vector<std::vector<std::uint8_t>>& lens_root,
                                                const std::vector<std::vector<std::uint8_t>>& lens_linkage,
                                                const std::vector<std::vector<std::uint8_t>>& payload_root,
                                                const std::vector<std::vector<std::uint8_t>>& payload_linkage,
                                                std::string* err) {
    if (cid < 0 || cid >= meta_.nlist) {
        if (err) *err = "LinkageCoeffCodecStore: cid out of range.";
        return false;
    }
    if (static_cast<int>(scales_root.size()) != meta_.m ||
        static_cast<int>(scales_linkage.size()) != meta_.m) {
        if (err) *err = "LinkageCoeffCodecStore: scales size mismatch.";
        return false;
    }

    const int streams = meta_.streams;
    const int S = meta_.S;
    if (static_cast<int>(lens_root.size()) != streams ||
        static_cast<int>(lens_linkage.size()) != streams ||
        static_cast<int>(payload_root.size()) != streams ||
        static_cast<int>(payload_linkage.size()) != streams) {
        if (err) *err = "LinkageCoeffCodecStore: stream count mismatch.";
        return false;
    }
    for (int s = 0; s < streams; ++s) {
        if (static_cast<int>(lens_root[static_cast<std::size_t>(s)].size()) != S ||
            static_cast<int>(lens_linkage[static_cast<std::size_t>(s)].size()) != S) {
            if (err) *err = "LinkageCoeffCodecStore: lens size mismatch.";
            return false;
        }
    }

    // Write scales (fixed-size, per cluster).
    {
        std::vector<float> tmp;
        tmp.reserve(static_cast<std::size_t>(2 * meta_.m));
        tmp.insert(tmp.end(), scales_root.begin(), scales_root.end());
        tmp.insert(tmp.end(), scales_linkage.begin(), scales_linkage.end());
        const std::uint64_t off = static_cast<std::uint64_t>(cid) * 2ull *
                                  static_cast<std::uint64_t>(meta_.m) * sizeof(float);
#if defined(_WIN32)
        std::lock_guard<std::mutex> lock(mu_scales_);
        scales_out_.seekp(static_cast<std::streamoff>(off), std::ios::beg);
        scales_out_.write(reinterpret_cast<const char*>(tmp.data()),
                          static_cast<std::streamsize>(tmp.size() * sizeof(float)));
        if (!scales_out_) {
            if (err) *err = "LinkageCoeffCodecStore: scales write failed.";
            return false;
        }
#else
        if (!PwriteAll(fd_scales_, tmp.data(), tmp.size() * sizeof(float), off, err)) {
            return false;
        }
#endif
    }

    // Fixed lens writes + variable payload writes (atomic append).
    auto write_group = [&](int group,
                           const std::vector<std::vector<std::uint8_t>>& lens,
                           const std::vector<std::vector<std::uint8_t>>& payloads) -> bool {
        for (int s = 0; s < streams; ++s) {
            const std::size_t idx =
                (static_cast<std::size_t>(group) * static_cast<std::size_t>(meta_.nlist) +
                 static_cast<std::size_t>(cid)) *
                    static_cast<std::size_t>(streams) +
                static_cast<std::size_t>(s);

            // Lens at deterministic offset.
            const std::uint64_t off_lens = static_cast<std::uint64_t>(idx) *
                                           static_cast<std::uint64_t>(S);
#if defined(_WIN32)
            {
                std::lock_guard<std::mutex> lock(mu_lens_);
                lens_out_.seekp(static_cast<std::streamoff>(off_lens), std::ios::beg);
                lens_out_.write(reinterpret_cast<const char*>(lens[static_cast<std::size_t>(s)].data()),
                                static_cast<std::streamsize>(S));
                if (!lens_out_) {
                    if (err) *err = "LinkageCoeffCodecStore: lens write failed.";
                    return false;
                }
            }
#else
            if (!PwriteAll(fd_lens_,
                           lens[static_cast<std::size_t>(s)].data(),
                           static_cast<std::size_t>(S),
                           off_lens,
                           err)) {
                return false;
            }
#endif

            // Payload: append anywhere, record (offset,size).
            const std::vector<std::uint8_t>& payload = payloads[static_cast<std::size_t>(s)];
            const auto sz = static_cast<std::uint32_t>(payload.size());
            const std::uint64_t off_payload = payload_cursor_.fetch_add(sz, std::memory_order_relaxed);
            payload_offs_u64_[idx] = off_payload;
            payload_sizes_u32_[idx] = sz;
            if (sz > 0) {
#if defined(_WIN32)
                std::lock_guard<std::mutex> lock(mu_payload_);
                payload_out_.seekp(static_cast<std::streamoff>(off_payload), std::ios::beg);
                payload_out_.write(reinterpret_cast<const char*>(payload.data()),
                                   static_cast<std::streamsize>(payload.size()));
                if (!payload_out_) {
                    if (err) *err = "LinkageCoeffCodecStore: payload write failed.";
                    return false;
                }
#else
                if (!PwriteAll(fd_payload_, payload.data(), payload.size(), off_payload, err)) {
                    return false;
                }
#endif
            }

            // Persist offsets/sizes at deterministic locations so the store can be resumed.
            const std::uint64_t off_offs = static_cast<std::uint64_t>(idx) * sizeof(std::uint64_t);
            const std::uint64_t off_sizes = static_cast<std::uint64_t>(idx) * sizeof(std::uint32_t);
#if defined(_WIN32)
            {
                std::lock_guard<std::mutex> lock(mu_offs_);
                payload_offs_out_.seekp(static_cast<std::streamoff>(off_offs), std::ios::beg);
                payload_offs_out_.write(reinterpret_cast<const char*>(&off_payload),
                                        static_cast<std::streamsize>(sizeof(std::uint64_t)));
                payload_sizes_out_.seekp(static_cast<std::streamoff>(off_sizes), std::ios::beg);
                payload_sizes_out_.write(reinterpret_cast<const char*>(&sz),
                                         static_cast<std::streamsize>(sizeof(std::uint32_t)));
                if (!payload_offs_out_ || !payload_sizes_out_) {
                    if (err) *err = "LinkageCoeffCodecStore: offsets/sizes write failed.";
                    return false;
                }
            }
#else
            if (!PwriteAll(fd_payload_offs_, &off_payload, sizeof(std::uint64_t), off_offs, err)) {
                return false;
            }
            if (!PwriteAll(fd_payload_sizes_, &sz, sizeof(std::uint32_t), off_sizes, err)) {
                return false;
            }
#endif
        }
        return true;
    };

    if (!write_group(/*group=*/0, lens_root, payload_root)) return false;
    if (!write_group(/*group=*/1, lens_linkage, payload_linkage)) return false;
    return true;
}

bool LinkageCoeffCodecRandomWriter::Finish(std::string* err) {
    (void)err;
    stats_.payload_bytes = payload_cursor_.load(std::memory_order_relaxed);

#if defined(_WIN32)
    payload_out_.close();
    lens_out_.close();
    scales_out_.close();
    payload_offs_out_.close();
    payload_sizes_out_.close();
#else
    if (fd_payload_ >= 0) ::close(fd_payload_);
    if (fd_lens_ >= 0) ::close(fd_lens_);
    if (fd_scales_ >= 0) ::close(fd_scales_);
    if (fd_payload_offs_ >= 0) ::close(fd_payload_offs_);
    if (fd_payload_sizes_ >= 0) ::close(fd_payload_sizes_);
    fd_payload_ = fd_lens_ = fd_scales_ = -1;
    fd_payload_offs_ = fd_payload_sizes_ = -1;
#endif
    return true;
}

bool LinkageCoeffCodecReader::Open(const std::string& dir, std::string* err) {
    dir_ = dir;
    if (dir_.empty()) {
        if (err) *err = "LinkageCoeffCodecReader: empty dir.";
        return false;
    }

    const std::string meta_path = JoinPath(dir_, "coeff_meta.bin");
    std::ifstream in(meta_path, std::ios::binary);
    if (!in.is_open()) {
        if (err) *err = "LinkageCoeffCodecReader: failed to open meta: " + meta_path;
        return false;
    }
    std::uint32_t magic = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != 0x51434343u) {
        if (err) *err = "LinkageCoeffCodecReader: bad magic.";
        return false;
    }
    in.read(reinterpret_cast<char*>(&meta_.version), sizeof(meta_.version));
    in.read(reinterpret_cast<char*>(&meta_.nlist), sizeof(meta_.nlist));
    in.read(reinterpret_cast<char*>(&meta_.m), sizeof(meta_.m));
    in.read(reinterpret_cast<char*>(&meta_.streams), sizeof(meta_.streams));
    in.read(reinterpret_cast<char*>(&meta_.max_bits), sizeof(meta_.max_bits));
    in.read(reinterpret_cast<char*>(&meta_.alphabet_size), sizeof(meta_.alphabet_size));
    in.read(reinterpret_cast<char*>(&meta_.S), sizeof(meta_.S));
    if (!in) {
        if (err) *err = "LinkageCoeffCodecReader: meta read failed.";
        return false;
    }
    if (meta_.nlist <= 0 || meta_.m <= 0 || meta_.streams <= 0 || meta_.S <= 0) {
        if (err) *err = "LinkageCoeffCodecReader: invalid meta.";
        return false;
    }

    if (!ReadBinaryFile(JoinPath(dir_, "coeff_payload_offs.u64"), &payload_offs_u64_, err)) {
        return false;
    }
    if (!ReadBinaryFile(JoinPath(dir_, "coeff_payload_sizes.u32"), &payload_sizes_u32_, err)) {
        return false;
    }
    const std::size_t expect = static_cast<std::size_t>(meta_.nlist) * 2u *
                               static_cast<std::size_t>(meta_.streams);
    if (payload_offs_u64_.size() != expect || payload_sizes_u32_.size() != expect) {
        if (err) *err = "LinkageCoeffCodecReader: offsets/sizes length mismatch.";
        return false;
    }
    return true;
}

bool LinkageCoeffCodecReader::ReadCluster(int cid,
                                       std::vector<float>* scales_root,
                                       std::vector<float>* scales_linkage,
                                       std::vector<std::vector<std::uint8_t>>* lens_root,
                                       std::vector<std::vector<std::uint8_t>>* lens_linkage,
                                       std::vector<std::vector<std::uint8_t>>* payload_root,
                                       std::vector<std::vector<std::uint8_t>>* payload_linkage,
                                       std::string* err) const {
    if (cid < 0 || cid >= meta_.nlist) {
        if (err) *err = "LinkageCoeffCodecReader: cid out of range.";
        return false;
    }

    const int streams = meta_.streams;
    const int S = meta_.S;

    scales_root->assign(static_cast<std::size_t>(meta_.m), 1.0f);
    scales_linkage->assign(static_cast<std::size_t>(meta_.m), 1.0f);
    lens_root->assign(static_cast<std::size_t>(streams), std::vector<std::uint8_t>(static_cast<std::size_t>(S), 0));
    lens_linkage->assign(static_cast<std::size_t>(streams), std::vector<std::uint8_t>(static_cast<std::size_t>(S), 0));
    payload_root->assign(static_cast<std::size_t>(streams), {});
    payload_linkage->assign(static_cast<std::size_t>(streams), {});

    // Read scales.
    {
        std::ifstream in(JoinPath(dir_, "coeff_scales.f32"), std::ios::binary);
        if (!in.is_open()) {
            if (err) *err = "LinkageCoeffCodecReader: missing scales file.";
            return false;
        }
        const std::uint64_t off = static_cast<std::uint64_t>(cid) * 2ull *
                                  static_cast<std::uint64_t>(meta_.m) * sizeof(float);
        in.seekg(static_cast<std::streamoff>(off), std::ios::beg);
        std::vector<float> tmp(static_cast<std::size_t>(2 * meta_.m));
        in.read(reinterpret_cast<char*>(tmp.data()),
                static_cast<std::streamsize>(tmp.size() * sizeof(float)));
        if (!in) {
            if (err) *err = "LinkageCoeffCodecReader: scales read failed.";
            return false;
        }
        std::copy(tmp.begin(), tmp.begin() + meta_.m, scales_root->begin());
        std::copy(tmp.begin() + meta_.m, tmp.end(), scales_linkage->begin());
    }

    // Read lens and payloads per stream.
    std::ifstream lens_in(JoinPath(dir_, "coeff_lens.u8"), std::ios::binary);
    if (!lens_in.is_open()) {
        if (err) *err = "LinkageCoeffCodecReader: missing lens file.";
        return false;
    }
    std::ifstream payload_in(JoinPath(dir_, "coeff_payload.bin"), std::ios::binary);
    if (!payload_in.is_open()) {
        if (err) *err = "LinkageCoeffCodecReader: missing payload file.";
        return false;
    }

    auto load_group = [&](int group,
                          std::vector<std::vector<std::uint8_t>>* lens_out,
                          std::vector<std::vector<std::uint8_t>>* payload_out) -> bool {
        for (int s = 0; s < streams; ++s) {
            const std::size_t idx =
                (static_cast<std::size_t>(group) * static_cast<std::size_t>(meta_.nlist) +
                 static_cast<std::size_t>(cid)) *
                    static_cast<std::size_t>(streams) +
                static_cast<std::size_t>(s);

            const std::uint64_t off_lens = static_cast<std::uint64_t>(idx) *
                                           static_cast<std::uint64_t>(S);
            lens_in.seekg(static_cast<std::streamoff>(off_lens), std::ios::beg);
            lens_out->at(static_cast<std::size_t>(s)).resize(static_cast<std::size_t>(S));
            lens_in.read(reinterpret_cast<char*>(lens_out->at(static_cast<std::size_t>(s)).data()),
                         static_cast<std::streamsize>(S));
            if (!lens_in) {
                if (err) *err = "LinkageCoeffCodecReader: lens read failed.";
                return false;
            }

            const std::uint64_t off_payload = payload_offs_u64_[idx];
            const std::uint32_t sz = payload_sizes_u32_[idx];
            payload_out->at(static_cast<std::size_t>(s)).resize(sz);
            if (sz > 0) {
                payload_in.seekg(static_cast<std::streamoff>(off_payload), std::ios::beg);
                payload_in.read(reinterpret_cast<char*>(payload_out->at(static_cast<std::size_t>(s)).data()),
                                static_cast<std::streamsize>(sz));
                if (!payload_in) {
                    if (err) *err = "LinkageCoeffCodecReader: payload read failed.";
                    return false;
                }
            }
        }
        return true;
    };

    if (!load_group(0, lens_root, payload_root)) return false;
    if (!load_group(1, lens_linkage, payload_linkage)) return false;
    return true;
}

}  // namespace stlq::io
