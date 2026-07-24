#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "stlq/common/types.h"

namespace stlq::ivf {

// Disk-cached HNSW index over IVF root centroids for eval probe selection.
//
// Space: inner-product over unit-normalized centroids.
// This matches the existing exact probe scoring used by linkage disk IVF eval:
//   score(cid) = dot(q, c) / ||c||  == dot(q, unit(c))
//
// Layout:
//   <index_dir>/index.bin   (serialized internal HNSW index)
//   <index_dir>/meta.txt    (key=value lines)
//
// Meta mismatch behavior: automatically rebuild and overwrite cache (run-tag output dirs already isolate runs).
class IvfProbeHnswIndex {
public:
    IvfProbeHnswIndex();
    ~IvfProbeHnswIndex();

    static std::filesystem::path DefaultIndexDirFromLinkageListDir(const std::filesystem::path& linkage_list_dir);

    // Load an existing index if present and compatible, otherwise build it from `C_root0`.
    //
    // Required invariants:
    // - d > 0, nlist > 0
    // - C_root0 is (d x nlist)
    bool LoadOrBuildUnitIP(const ColMajorMatrix<float>& C_root0,
                           const std::filesystem::path& index_dir,
                           int M,
                           int ef_construction,
                           int build_threads,
                           bool* out_built,
                           std::string* err);

    // Set runtime `ef_search` used by subsequent SearchTopK calls.
    // NOTE: this mutates underlying HNSW search state; call it outside parallel regions.
    bool SetEfSearch(int ef_search, std::string* err);

    // Search for top `k` centroids (nearest by IP distance) and write results.
    // - out_ids: centroid ids in [0, nlist)
    // - out_dist: IP distance (1 - dot(q, unit(c))) ascending (nearest first). Optional.
    bool SearchTopK(const float* query,
                    int k,
                    int* out_ids,
                    float* out_dist,
                    std::string* err) const;

    int d() const { return d_; }
    int nlist() const { return nlist_; }

private:
    int d_ = 0;
    int nlist_ = 0;
    int M_ = 0;
    int ef_construction_ = 0;
    std::uint64_t centroids_unit_hash_ = 0;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace stlq::ivf
