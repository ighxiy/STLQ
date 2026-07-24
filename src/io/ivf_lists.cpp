#include "stlq/io/ivf_lists.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <list>
#include <limits>
#include <cstring>
#include <string>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
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
        if (err) {
            *err = "IvfLists: failed to create dir: " + dir;
        }
        return false;
    }
}

bool ReadU32Block(std::ifstream& in, std::vector<std::uint32_t>* buf, std::size_t max_elems) {
    buf->resize(max_elems);
    in.read(reinterpret_cast<char*>(buf->data()),
            static_cast<std::streamsize>(max_elems * sizeof(std::uint32_t)));
    const std::streamsize got = in.gcount();
    if (got <= 0) {
        buf->clear();
        return false;
    }
    const std::size_t elems = static_cast<std::size_t>(got) / sizeof(std::uint32_t);
    buf->resize(elems);
    return true;
}

int DetermineMaxOpenBuckets(int nbucket) {
    // Keep some headroom for other files/sockets.
    int max_open = 256;
#if defined(__unix__) || defined(__APPLE__)
    struct rlimit rl;
    if (getrlimit(RLIMIT_NOFILE, &rl) == 0) {
        // rlim_cur may be RLIM_INFINITY.
        const auto cur = rl.rlim_cur;
        if (cur != RLIM_INFINITY) {
            const long safe = static_cast<long>(cur) - 128;
            if (safe > 0) {
                max_open = static_cast<int>(safe);
            }
        } else {
            max_open = 2048;
        }
    }
#endif
    if (max_open < 32) max_open = 32;
    if (max_open > 2048) max_open = 2048;
    if (max_open > nbucket) max_open = nbucket;
    return max_open;
}

}  // namespace

bool BuildIvfListsFromClusterIdFile(const std::string& cluster_id_u32,
                                   int nlist,
                                   int bucket_size,
                                   const IvfListsPaths& out,
                                   const std::string& tmp_dir,
                                   bool keep_tmp,
                                   std::string* err) {
    if (nlist <= 0 || bucket_size <= 0) {
        if (err) {
            *err = "BuildIvfListsFromClusterIdFile: invalid nlist/bucket_size.";
        }
        return false;
    }
    if (cluster_id_u32.empty() || out.offsets_u64.empty() || out.ids_u32.empty() ||
        tmp_dir.empty()) {
        if (err) {
            *err = "BuildIvfListsFromClusterIdFile: invalid paths.";
        }
        return false;
    }
    if (!EnsureDir(std::filesystem::path(out.offsets_u64).parent_path().string(), err)) {
        return false;
    }
    if (!EnsureDir(std::filesystem::path(out.ids_u32).parent_path().string(), err)) {
        return false;
    }
    if (!EnsureDir(tmp_dir, err)) {
        return false;
    }

    const int nbucket = (nlist + bucket_size - 1) / bucket_size;

    // Pass 1: scan cluster_id.u32 and write bucketed (local_cid_u16, gid_u32) pairs.
    std::vector<std::uint64_t> counts(static_cast<std::size_t>(nlist), 0);
    std::vector<std::ofstream> bucket_out(static_cast<std::size_t>(nbucket));
    std::vector<bool> bucket_open(static_cast<std::size_t>(nbucket), false);
    std::vector<bool> bucket_created(static_cast<std::size_t>(nbucket), false);

    const int max_open_buckets = DetermineMaxOpenBuckets(nbucket);
    int open_count = 0;
    std::list<int> lru;
    std::vector<std::list<int>::iterator> lru_it(static_cast<std::size_t>(nbucket), lru.end());

    auto lru_touch = [&](int b) {
        auto& it = lru_it[static_cast<std::size_t>(b)];
        if (it == lru.end()) {
            lru.push_back(b);
            it = std::prev(lru.end());
        } else {
            lru.splice(lru.end(), lru, it);
        }
    };

    auto close_bucket = [&](int b) {
        const std::size_t bi = static_cast<std::size_t>(b);
        if (bucket_open[bi]) {
            if (bucket_out[bi].is_open()) {
                bucket_out[bi].close();
            }
            bucket_open[bi] = false;
            if (open_count > 0) --open_count;
        }
        auto& it = lru_it[bi];
        if (it != lru.end()) {
            lru.erase(it);
            it = lru.end();
        }
    };

    auto ensure_bucket = [&](int b) -> bool {
        const std::size_t bi = static_cast<std::size_t>(b);
        if (bucket_open[bi]) {
            lru_touch(b);
            return true;
        }

        while (open_count >= max_open_buckets && !lru.empty()) {
            close_bucket(lru.front());
        }

        const std::string path = JoinPath(tmp_dir, "pairs_bucket_" + std::to_string(b) + ".bin");
        const auto mode = std::ios::binary | (bucket_created[bi] ? std::ios::app : std::ios::trunc);
        bucket_out[bi].open(path, mode);
        if (!bucket_out[bi].is_open()) {
            if (err) {
                *err = "IvfLists: failed to open tmp bucket file: " + path +
                       " (errno=" + std::to_string(errno) + ": " + std::string(std::strerror(errno)) + ")";
            }
            return false;
        }
        bucket_created[bi] = true;
        bucket_open[bi] = true;
        ++open_count;
        lru_touch(b);
        return true;
    };

    std::ifstream in(cluster_id_u32, std::ios::binary);
    if (!in.is_open()) {
        if (err) {
            *err = "IvfLists: failed to open cluster_id.u32: " + cluster_id_u32;
        }
        return false;
    }

#pragma pack(push, 1)
    struct Pair {
        std::uint16_t local;
        std::uint32_t gid;
    };
#pragma pack(pop)
    std::vector<std::vector<Pair>> bucket_bufs(static_cast<std::size_t>(nbucket));
    // Provide a decent buffer size per bucket (e.g. 256KB = ~43690 pairs) to amortize open/close.
    // 3907 buckets * 256KB ~ 1GB total memory overhead, highly effective at avoiding FD thrashing.
    const std::size_t buf_capacity = 43690;
    for (auto& bbuf : bucket_bufs) {
        bbuf.reserve(buf_capacity);
    }

    auto flush_bucket_buf = [&](int b) -> bool {
        const std::size_t bi = static_cast<std::size_t>(b);
        if (bucket_bufs[bi].empty()) return true;
        if (!ensure_bucket(b)) return false;
        bucket_out[bi].write(reinterpret_cast<const char*>(bucket_bufs[bi].data()),
                             static_cast<std::streamsize>(bucket_bufs[bi].size() * sizeof(Pair)));
        if (!bucket_out[bi]) {
            if (err) {
                *err = "IvfLists: failed while writing tmp pairs.";
            }
            return false;
        }
        bucket_bufs[bi].clear();
        return true;
    };

    std::vector<std::uint32_t> buf;
    buf.reserve(1u << 20);
    std::uint64_t gid = 0;
    const std::size_t block_elems = 1u << 20;
    while (ReadU32Block(in, &buf, block_elems)) {
        for (std::size_t i = 0; i < buf.size(); ++i, ++gid) {
            const std::uint32_t cid_u32 = buf[i];
            const int cid = static_cast<int>(cid_u32);
            if (cid < 0 || cid >= nlist) {
                if (err) {
                    *err = "IvfLists: cluster id out of range in cluster_id file.";
                }
                return false;
            }
            ++counts[static_cast<std::size_t>(cid)];

            const int b = cid / bucket_size;
            const auto local = static_cast<std::uint16_t>(cid - b * bucket_size);
            const std::size_t bi = static_cast<std::size_t>(b);
            bucket_bufs[bi].push_back({local, static_cast<std::uint32_t>(gid)});

            if (bucket_bufs[bi].size() >= buf_capacity) {
                if (!flush_bucket_buf(b)) {
                    return false;
                }
            }
        }
    }

    // Flush any remaining buffered pairs
    for (int b = 0; b < nbucket; ++b) {
        if (!flush_bucket_buf(b)) {
            return false;
        }
    }

    // Reuse the chunk buffer across buckets to avoid repeated allocations/zero-fill.
    std::vector<std::uint8_t> pbuf(6u * (1u << 20));

    for (int b = 0; b < nbucket; ++b) {
        close_bucket(b);
    }

    // Build offsets and write offsets file.
    std::vector<std::uint64_t> offsets(static_cast<std::size_t>(nlist) + 1, 0);
    for (int c = 0; c < nlist; ++c) {
        offsets[static_cast<std::size_t>(c + 1)] =
            offsets[static_cast<std::size_t>(c)] + counts[static_cast<std::size_t>(c)];
    }
    {
        std::ofstream off(out.offsets_u64, std::ios::binary | std::ios::trunc);
        if (!off.is_open()) {
            if (err) {
                *err = "IvfLists: failed to open offsets file: " + out.offsets_u64;
            }
            return false;
        }
        off.write(reinterpret_cast<const char*>(offsets.data()),
                  static_cast<std::streamsize>(offsets.size() * sizeof(std::uint64_t)));
        if (!off) {
            if (err) {
                *err = "IvfLists: failed to write offsets file.";
            }
            return false;
        }
    }

    // Pass 2: for each bucket, read tmp pairs and place gids into a flat buffer (stable per-cluster order),
    // then write the whole bucket segment in one contiguous write.
    std::ofstream ids(out.ids_u32, std::ios::binary | std::ios::trunc);
    if (!ids.is_open()) {
        if (err) {
            *err = "IvfLists: failed to open ids file: " + out.ids_u32;
        }
        return false;
    }

    for (int b = 0; b < nbucket; ++b) {
        const int c0 = b * bucket_size;
        const int c1 = std::min(nlist, (b + 1) * bucket_size);
        const int nlocal = c1 - c0;

        std::uint64_t total_u64 = 0;
        std::vector<std::uint64_t> local_counts(static_cast<std::size_t>(nlocal), 0);
        for (int lc = 0; lc < nlocal; ++lc) {
            const int cid = c0 + lc;
            const std::uint64_t cnt = counts[static_cast<std::size_t>(cid)];
            local_counts[static_cast<std::size_t>(lc)] = cnt;
            total_u64 += cnt;
        }
        if (total_u64 == 0) {
            continue;
        }
        if (total_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            if (err) {
                *err = "IvfLists: bucket too large to allocate buffer.";
            }
            return false;
        }
        const std::size_t total = static_cast<std::size_t>(total_u64);

        std::vector<std::size_t> local_off(static_cast<std::size_t>(nlocal) + 1, 0);
        for (int lc = 0; lc < nlocal; ++lc) {
            const auto cnt = static_cast<std::size_t>(
                std::min<std::uint64_t>(local_counts[static_cast<std::size_t>(lc)],
                                        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())));
            local_off[static_cast<std::size_t>(lc + 1)] = local_off[static_cast<std::size_t>(lc)] + cnt;
        }
        if (local_off.back() != total) {
            if (err) {
                *err = "IvfLists: internal offset size mismatch.";
            }
            return false;
        }

        std::vector<std::size_t> cursor(static_cast<std::size_t>(nlocal), 0);
        std::vector<std::uint32_t> bucket_ids(total);

        const std::string path = JoinPath(tmp_dir, "pairs_bucket_" + std::to_string(b) + ".bin");
        std::ifstream pb(path, std::ios::binary);
        if (!pb.is_open()) {
            // Should not happen when total>0, but tolerate missing file as empty.
            continue;
        }

        while (true) {
            pb.read(reinterpret_cast<char*>(pbuf.data()), static_cast<std::streamsize>(pbuf.size()));
            const std::streamsize got = pb.gcount();
            if (got <= 0) {
                break;
            }
            const auto bytes = static_cast<std::size_t>(got);
            const std::size_t npair = bytes / 6u;
            const std::uint8_t* ptr = pbuf.data();
            for (std::size_t i = 0; i < npair; ++i) {
                std::uint16_t local = 0;
                std::uint32_t gid32 = 0;
                std::memcpy(&local, ptr, sizeof(std::uint16_t));
                std::memcpy(&gid32, ptr + sizeof(std::uint16_t), sizeof(std::uint32_t));
                ptr += 6u;
                if (local >= static_cast<std::uint16_t>(nlocal)) {
                    if (err) {
                        *err = "IvfLists: local cluster id out of range in tmp pairs.";
                    }
                    return false;
                }
                const std::size_t li = static_cast<std::size_t>(local);
                const std::size_t pos = local_off[li] + cursor[li];
                if (pos >= total) {
                    if (err) {
                        *err = "IvfLists: tmp pairs overflow bucket buffer.";
                    }
                    return false;
                }
                bucket_ids[pos] = gid32;
                ++cursor[li];
            }
        }

        for (int lc = 0; lc < nlocal; ++lc) {
            if (cursor[static_cast<std::size_t>(lc)] != static_cast<std::size_t>(
                    std::min<std::uint64_t>(local_counts[static_cast<std::size_t>(lc)],
                                            static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())))) {
                if (err) {
                    *err = "IvfLists: tmp pairs count mismatch for a local cluster.";
                }
                return false;
            }
        }

        const std::uint64_t base_off_u64 = offsets[static_cast<std::size_t>(c0)];
        if (base_off_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max() / sizeof(std::uint32_t))) {
            if (err) {
                *err = "IvfLists: output offset overflow.";
            }
            return false;
        }
        ids.seekp(static_cast<std::streamoff>(base_off_u64 * sizeof(std::uint32_t)), std::ios::beg);
        ids.write(reinterpret_cast<const char*>(bucket_ids.data()),
                  static_cast<std::streamsize>(bucket_ids.size() * sizeof(std::uint32_t)));
        if (!ids) {
            if (err) {
                *err = "IvfLists: failed to write ids file.";
            }
            return false;
        }
    }

    ids.flush();
    if (!ids) {
        if (err) {
            *err = "IvfLists: failed to flush ids file.";
        }
        return false;
    }

    if (!keep_tmp) {
        std::error_code ec;
        std::filesystem::remove_all(tmp_dir, ec);
    }

    return true;
}

bool IvfListsReader::Open(const IvfListsPaths& paths, std::string* err) {
    paths_ = paths;
    offsets_.clear();
    nlist_ = 0;
    ntotal_ = 0;

    if (paths_.offsets_u64.empty() || paths_.ids_u32.empty()) {
        if (err) {
            *err = "IvfListsReader::Open: empty paths.";
        }
        return false;
    }
    std::ifstream off(paths_.offsets_u64, std::ios::binary);
    if (!off.is_open()) {
        if (err) {
            *err = "IvfListsReader::Open: failed to open offsets file.";
        }
        return false;
    }
    off.seekg(0, std::ios::end);
    const std::streamoff sz = off.tellg();
    if (sz <= 0 || (sz % static_cast<std::streamoff>(sizeof(std::uint64_t))) != 0) {
        if (err) {
            *err = "IvfListsReader::Open: invalid offsets size.";
        }
        return false;
    }
    const std::size_t n = static_cast<std::size_t>(sz) / sizeof(std::uint64_t);
    offsets_.resize(n);
    off.seekg(0, std::ios::beg);
    off.read(reinterpret_cast<char*>(offsets_.data()),
             static_cast<std::streamsize>(offsets_.size() * sizeof(std::uint64_t)));
    if (!off) {
        if (err) {
            *err = "IvfListsReader::Open: failed to read offsets.";
        }
        return false;
    }
    if (offsets_.size() < 2) {
        if (err) {
            *err = "IvfListsReader::Open: offsets too small.";
        }
        return false;
    }
    nlist_ = static_cast<int>(offsets_.size() - 1);
    ntotal_ = offsets_.back();
    return true;
}

bool IvfListsReader::ReadList(int cid, std::vector<std::uint32_t>* out, std::string* err) const {
    if (!out) {
        if (err) {
            *err = "IvfListsReader::ReadList: null output.";
        }
        return false;
    }
    out->clear();
    if (cid < 0 || cid >= nlist_) {
        return true;
    }
    const std::uint64_t begin = offsets_[static_cast<std::size_t>(cid)];
    const std::uint64_t end = offsets_[static_cast<std::size_t>(cid + 1)];
    if (end <= begin) {
        return true;
    }
    const std::uint64_t len64 = end - begin;
    if (len64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        if (err) {
            *err = "IvfListsReader::ReadList: list too large for host memory.";
        }
        return false;
    }
    const auto len = static_cast<std::size_t>(len64);
    out->resize(len);

    std::ifstream ids(paths_.ids_u32, std::ios::binary);
    if (!ids.is_open()) {
        if (err) {
            *err = "IvfListsReader::ReadList: failed to open ids file.";
        }
        return false;
    }
    const std::uint64_t off_bytes = begin * sizeof(std::uint32_t);
    ids.seekg(static_cast<std::streamoff>(off_bytes), std::ios::beg);
    if (!ids) {
        if (err) {
            *err = "IvfListsReader::ReadList: seek failed.";
        }
        return false;
    }
    ids.read(reinterpret_cast<char*>(out->data()),
             static_cast<std::streamsize>(len * sizeof(std::uint32_t)));
    if (!ids) {
        if (err) {
            *err = "IvfListsReader::ReadList: read failed.";
        }
        return false;
    }
    return true;
}

IvfListsThreadReader::~IvfListsThreadReader() {
    Close();
}

void IvfListsThreadReader::Close() {
    if (ids_in_.is_open()) {
        ids_in_.close();
    }
    ids_path_.clear();
}

bool IvfListsThreadReader::OpenFrom(const IvfListsReader& src, std::string* err) {
    Close();
    ids_path_ = src.paths().ids_u32;
    ids_in_.open(ids_path_, std::ios::binary);
    if (!ids_in_.is_open()) {
        if (err) {
            *err = "IvfListsThreadReader::Open: failed to open ids file.";
        }
        return false;
    }
    return true;
}

bool IvfListsThreadReader::ReadList(const IvfListsReader& src,
                                    int cid,
                                    std::vector<std::uint32_t>* out,
                                    std::string* err) {
    if (!out) {
        if (err) {
            *err = "IvfListsThreadReader::ReadList: null output.";
        }
        return false;
    }
    out->clear();
    if (!ids_in_.is_open() || ids_path_ != src.paths().ids_u32) {
        if (!OpenFrom(src, err)) {
            return false;
        }
    }
    if (cid < 0 || cid >= src.nlist()) {
        return true;
    }
    const std::uint64_t begin = src.offsets()[static_cast<std::size_t>(cid)];
    const std::uint64_t end = src.offsets()[static_cast<std::size_t>(cid + 1)];
    if (end <= begin) {
        return true;
    }
    const std::uint64_t len64 = end - begin;
    if (len64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        if (err) {
            *err = "IvfListsThreadReader::ReadList: list too large for host memory.";
        }
        return false;
    }
    const auto len = static_cast<std::size_t>(len64);
    out->resize(len);
    const std::uint64_t off_bytes = begin * sizeof(std::uint32_t);
    ids_in_.seekg(static_cast<std::streamoff>(off_bytes), std::ios::beg);
    if (!ids_in_) {
        if (err) {
            *err = "IvfListsThreadReader::ReadList: seek failed.";
        }
        return false;
    }
    ids_in_.read(reinterpret_cast<char*>(out->data()),
                 static_cast<std::streamsize>(len * sizeof(std::uint32_t)));
    if (!ids_in_) {
        if (err) {
            *err = "IvfListsThreadReader::ReadList: read failed.";
        }
        return false;
    }
    return true;
}

}  // namespace stlq::io
