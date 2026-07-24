#pragma once

#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace stlq::io {

// Disk IVF lists in CSR form:
// - offsets: length (nlist + 1), uint64, monotonically increasing
// - ids: length offsets.back(), uint32 global ids, concatenated by cluster
struct IvfListsPaths {
    std::string offsets_u64;
    std::string ids_u32;
};

// Build IVF CSR lists from per-vector root assignment file
// (`cluster_id.u32`, uint32 array of length=nbase).
// This implementation avoids random writes by using bucketed temporary files:
// - pass 1: write (local_cid_u16, gid_u32) pairs into per-bucket temp files, and count per cluster
// - pass 2: read each bucket temp file, group ids by local_cid, and write to `ids_u32` in cluster order
bool BuildIvfListsFromClusterIdFile(const std::string& cluster_id_u32,
                                   int nlist,
                                   int bucket_size,
                                   const IvfListsPaths& out,
                                   const std::string& tmp_dir,
                                   bool keep_tmp,
                                   std::string* err);

class IvfListsReader {
public:
    bool Open(const IvfListsPaths& paths, std::string* err);

    const IvfListsPaths& paths() const { return paths_; }
    int nlist() const { return nlist_; }
    std::uint64_t ntotal() const { return ntotal_; }
    const std::vector<std::uint64_t>& offsets() const { return offsets_; }

    std::uint64_t Offset(int cid) const {
        if (cid < 0 || cid >= nlist_) return 0;
        return offsets_[static_cast<std::size_t>(cid)];
    }
    std::uint32_t ListSize(int cid) const {
        if (cid < 0 || cid >= nlist_) return 0;
        const std::uint64_t b = offsets_[static_cast<std::size_t>(cid)];
        const std::uint64_t e = offsets_[static_cast<std::size_t>(cid + 1)];
        if (e <= b) return 0;
        const std::uint64_t len = e - b;
        if (len > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
            return static_cast<std::uint32_t>(std::numeric_limits<std::uint32_t>::max());
        }
        return static_cast<std::uint32_t>(len);
    }

    // Read ids for a given cluster into `out` (overwrites).
    bool ReadList(int cid, std::vector<std::uint32_t>* out, std::string* err) const;

private:
    IvfListsPaths paths_;
    int nlist_ = 0;
    std::uint64_t ntotal_ = 0;
    std::vector<std::uint64_t> offsets_;
};

// Thread-local (or single-thread) reader that keeps `ids_u32` open to avoid open/close per list.
// Not thread-safe for concurrent calls; create one instance per thread.
class IvfListsThreadReader {
public:
    IvfListsThreadReader() = default;
    ~IvfListsThreadReader();

    bool OpenFrom(const IvfListsReader& src, std::string* err);
    bool ReadList(const IvfListsReader& src, int cid, std::vector<std::uint32_t>* out, std::string* err);

private:
    void Close();

    std::string ids_path_;
    std::ifstream ids_in_;
};

}  // namespace stlq::io
