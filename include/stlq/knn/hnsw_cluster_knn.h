#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "stlq/common/config.h"
#include "stlq/common/types.h"

namespace stlq {

inline int ResolveInternalHnswEfConstruction(const HnswConfig& cfg,
                                             int requested_ef_construction,
                                             int max_elements) {
    int efc = std::max(1, requested_ef_construction);
    if (cfg.ef_construction_cap > 0) {
        efc = std::min(efc, cfg.ef_construction_cap);
    }
    if (max_elements > 0) {
        efc = std::min(efc, max_elements);
    }
    return std::max(1, efc);
}

// Per-cluster HNSW index wrapper:
// - labels are cluster-local ids [0..csize-1]
// - metric is squared L2
class ClusterHnswIndex {
public:
    ClusterHnswIndex();
    ~ClusterHnswIndex();

    ClusterHnswIndex(ClusterHnswIndex&&) noexcept;
    ClusterHnswIndex& operator=(ClusterHnswIndex&&) noexcept;

    ClusterHnswIndex(const ClusterHnswIndex&) = delete;
    ClusterHnswIndex& operator=(const ClusterHnswIndex&) = delete;

    void Build(const ColMajorMatrix<float>& X,
               const std::vector<int>& cluster_indices,
               const HnswConfig& cfg,
               int ef_construction);

    bool IsReady() const;
    int Size() const;

    void SetEf(int ef_search) const;

    // Writes labels only, sorted by ascending distance.
    void SearchLabelsInto(const float* query, int k, std::vector<int>* out) const;

    // Writes the nearest neighbors excluding `self_label`, sorted by ascending distance.
    void SearchNeighborsExcludingSelfInto(const float* query,
                                          int self_label,
                                          int k,
                                          std::vector<int>* out_labels,
                                          std::vector<float>* out_dists) const;

private:
    struct Impl;
    int size_ = 0;
    std::unique_ptr<Impl> impl_;
};

bool BuildClusterKnnHnswImpl(const stlq::ColMajorMatrix<float>& X,
                             const std::vector<int>& cols,
                             const stlq::HnswConfig& cfg,
                             int k_graph,
                             std::vector<std::uint32_t>* ids_flat,
                             std::vector<float>* dists_flat,
                             std::string* error);
}  // namespace stlq
