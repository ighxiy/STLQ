#include "stlq/knn/hnsw_cluster_knn.h"

#include "hnswlib/hnswlib.h"
#include "hnswlib/space_l2.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <memory>

namespace stlq {

struct ClusterHnswIndex::Impl {
    std::unique_ptr<hnswlib::L2Space> space;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> index;
};

ClusterHnswIndex::ClusterHnswIndex() = default;
ClusterHnswIndex::~ClusterHnswIndex() = default;

ClusterHnswIndex::ClusterHnswIndex(ClusterHnswIndex&&) noexcept = default;
ClusterHnswIndex& ClusterHnswIndex::operator=(ClusterHnswIndex&&) noexcept = default;

void ClusterHnswIndex::Build(const ColMajorMatrix<float>& X,
                             const std::vector<int>& cluster_indices,
                             const HnswConfig& cfg,
                             int ef_construction) {
    const int d = X.rows;
    const int csize = static_cast<int>(cluster_indices.size());
    const int max_elements = std::max(0, csize);
    size_ = max_elements;
    if (d <= 0 || max_elements <= 1) {
        return;
    }

    const int M = std::max(1, cfg.M);
    const int efc = ResolveInternalHnswEfConstruction(cfg, ef_construction, max_elements);
    auto next = std::make_unique<Impl>();
    next->space = std::make_unique<hnswlib::L2Space>(d);
    next->index = std::make_unique<hnswlib::HierarchicalNSW<float>>(next->space.get(),
                                                                    max_elements,
                                                                    M,
                                                                    efc);

    for (int local = 0; local < csize; ++local) {
        const int g = cluster_indices[static_cast<std::size_t>(local)];
        next->index->addPoint(X.Col(g), static_cast<hnswlib::labeltype>(local));
    }
    impl_ = std::move(next);
}

bool ClusterHnswIndex::IsReady() const {
    return impl_ && impl_->index != nullptr;
}

int ClusterHnswIndex::Size() const {
    return size_;
}

void ClusterHnswIndex::SetEf(int ef_search) const {
    if (IsReady()) {
        impl_->index->setEf(static_cast<std::size_t>(std::max(1, ef_search)));
    }
}

void ClusterHnswIndex::SearchLabelsInto(const float* query, int k, std::vector<int>* out) const {
    if (!IsReady()) {
        out->clear();
        return;
    }
    int k_use = std::max(0, k);
    if (size_ > 0) {
        k_use = std::min(k_use, size_);
    }
    if (k_use == 0) {
        out->clear();
        return;
    }

    auto pq = impl_->index->searchKnn(query, static_cast<std::size_t>(k_use));
    out->clear();
    out->reserve(pq.size());
    while (!pq.empty()) {
        out->push_back(static_cast<int>(pq.top().second));
        pq.pop();
    }
    std::reverse(out->begin(), out->end());
}

void ClusterHnswIndex::SearchNeighborsExcludingSelfInto(const float* query,
                                                        int self_label,
                                                        int k,
                                                        std::vector<int>* out_labels,
                                                        std::vector<float>* out_dists) const {
    if (!IsReady()) {
        out_labels->clear();
        if (out_dists) out_dists->clear();
        return;
    }
    int k_use = std::max(0, k);
    if (size_ > 0) {
        k_use = std::min(k_use, size_);
    }
    if (k_use == 0) {
        out_labels->clear();
        if (out_dists) out_dists->clear();
        return;
    }

    int want = k_use;
    if (self_label >= 0 && self_label < size_ && size_ > k_use) {
        want = std::min(size_, k_use + 1);
    }

    auto pq = impl_->index->searchKnn(query, static_cast<std::size_t>(want));
    out_labels->clear();
    out_labels->reserve(static_cast<std::size_t>(k_use));
    if (out_dists) {
        out_dists->clear();
        out_dists->reserve(static_cast<std::size_t>(k_use));
    }
    while (!pq.empty() && static_cast<int>(out_labels->size()) < k_use) {
        const float dist2 = pq.top().first;
        const int label = static_cast<int>(pq.top().second);
        pq.pop();
        if (label == self_label) {
            continue;
        }
        out_labels->push_back(label);
        if (out_dists) {
            out_dists->push_back(dist2);
        }
    }
    std::reverse(out_labels->begin(), out_labels->end());
    if (out_dists) {
        std::reverse(out_dists->begin(), out_dists->end());
    }
}

}  // namespace stlq

bool stlq::BuildClusterKnnHnswImpl(const stlq::ColMajorMatrix<float>& X,
                                     const std::vector<int>& cols,
                                     const stlq::HnswConfig& cfg,
                                     int k_graph,
                                     std::vector<std::uint32_t>* ids_flat,
                                     std::vector<float>* dists_flat,
                                     std::string* error) {
        if (!ids_flat) {
            if (error) {
                *error = "BuildClusterKnnHnswImpl: null ids output.";
            }
            return false;
        }
        const int d = X.rows;
        const int csize = static_cast<int>(cols.size());
        const int k_use = std::min(std::max(0, k_graph), std::max(0, csize - 1));

        ids_flat->resize(static_cast<std::size_t>(csize) * static_cast<std::size_t>(k_use));
        if (dists_flat) {
            dists_flat->resize(static_cast<std::size_t>(csize) * static_cast<std::size_t>(k_use));
        }
        if (d <= 0 || csize <= 1 || k_use <= 0) {
            return true;
        }

        // Build once per cluster and query all points. We keep label=local-id (0..csize-1).
        const int efc = std::max(1, std::min(k_use, csize));
        stlq::ClusterHnswIndex hnsw;
        hnsw.Build(X, cols, cfg, efc);
        if (!hnsw.IsReady()) {
            if (error) {
                *error = "BuildClusterKnnHnswImpl: HNSW index not ready.";
            }
            return false;
        }

        // Hot path: reuse per-thread buffers (this runs for all bad clusters / UMAP prep).
        static thread_local std::vector<int> tmp_labels;
        static thread_local std::vector<float> tmp_dists;
        if (tmp_labels.capacity() < static_cast<std::size_t>(k_use)) {
            tmp_labels.reserve(static_cast<std::size_t>(k_use));
        }
        if (dists_flat && tmp_dists.capacity() < static_cast<std::size_t>(k_use)) {
            tmp_dists.reserve(static_cast<std::size_t>(k_use));
        }
        hnsw.SetEf(std::min(csize, k_use + 1));
        for (int i = 0; i < csize; ++i) {
            const float* xi = X.Col(cols[static_cast<std::size_t>(i)]);

            try {
                hnsw.SearchNeighborsExcludingSelfInto(xi,
                                                      i,
                                                      k_use,
                                                      &tmp_labels,
                                                      dists_flat ? &tmp_dists : nullptr);
            } catch (const std::exception& e) {
                if (error) {
                    *error = std::string("BuildClusterKnnHnswImpl: searchKnn failed: ") + e.what();
                }
                return false;
            }

            std::uint32_t* out_ids_u32 =
                ids_flat->data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(k_use);
            float* out_dist = dists_flat ? (dists_flat->data() + static_cast<std::size_t>(i) * k_use)
                                         : nullptr;
            const int filled = static_cast<int>(tmp_labels.size());
            for (int j = 0; j < filled; ++j) {
                out_ids_u32[j] = static_cast<std::uint32_t>(tmp_labels[static_cast<std::size_t>(j)]);
                if (out_dist) {
                    out_dist[j] = tmp_dists[static_cast<std::size_t>(j)];
                }
            }
            if (filled < k_use) {
                if (error) {
                    *error = "BuildClusterKnnHnswImpl: insufficient neighbors.";
                }
                return false;
            }
        }
        return true;
    }
