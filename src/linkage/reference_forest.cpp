#include "stlq/linkage/reference_forest.h"

#include <algorithm>
#include <deque>
#include <limits>
#include <numeric>

#include "stlq/knn/hnsw_cluster_knn.h"

namespace stlq {
namespace {

std::uint64_t Mix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27U)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31U);
}

void CutCycles(std::vector<int>* parent, int* cuts) {
    const int n = static_cast<int>(parent->size());
    std::vector<std::uint8_t> state(static_cast<std::size_t>(n), 0);
    std::vector<int> path;
    path.reserve(static_cast<std::size_t>(n));

    for (int start = 0; start < n; ++start) {
        if (state[static_cast<std::size_t>(start)] != 0) continue;
        path.clear();
        int node = start;
        while (node >= 0 && state[static_cast<std::size_t>(node)] == 0) {
            state[static_cast<std::size_t>(node)] = 1;
            path.push_back(node);
            node = (*parent)[static_cast<std::size_t>(node)];
        }
        if (node >= 0 && state[static_cast<std::size_t>(node)] == 1) {
            auto first = std::find(path.begin(), path.end(), node);
            int cut = *first;
            for (auto it = first; it != path.end(); ++it) cut = std::min(cut, *it);
            (*parent)[static_cast<std::size_t>(cut)] = -1;
            ++(*cuts);
        }
        for (int v : path) state[static_cast<std::size_t>(v)] = 2;
    }
}

void BuildDepthLimitedOrder(int max_depth, ReferenceForest* forest) {
    const int n = static_cast<int>(forest->parent.size());
    std::vector<std::vector<int>> children(static_cast<std::size_t>(n));
    for (int child = 0; child < n; ++child) {
        const int p = forest->parent[static_cast<std::size_t>(child)];
        if (p >= 0) children[static_cast<std::size_t>(p)].push_back(child);
    }

    forest->depth.assign(static_cast<std::size_t>(n), 0);
    forest->order.clear();
    forest->order.reserve(static_cast<std::size_t>(n));
    std::deque<int> queue;
    for (int i = 0; i < n; ++i) {
        if (forest->parent[static_cast<std::size_t>(i)] < 0) queue.push_back(i);
    }

    while (!queue.empty()) {
        const int node = queue.front();
        queue.pop_front();
        forest->order.push_back(node);
        for (int child : children[static_cast<std::size_t>(node)]) {
            const int next_depth = forest->depth[static_cast<std::size_t>(node)] + 1;
            if (next_depth > max_depth) {
                forest->parent[static_cast<std::size_t>(child)] = -1;
                forest->depth[static_cast<std::size_t>(child)] = 0;
                ++forest->depth_cuts;
            } else {
                forest->depth[static_cast<std::size_t>(child)] = next_depth;
            }
            queue.push_back(child);
        }
    }
}

}  // namespace

bool BuildReferenceForest(const std::string& policy,
                          const ColMajorMatrix<float>& X,
                          const std::vector<int>& cluster_indices,
                          const HnswConfig& hnsw_cfg,
                          int max_depth,
                          std::uint64_t seed,
                          ReferenceForest* out,
                          std::string* error) {
    if (!out) {
        if (error) *error = "BuildReferenceForest: null output.";
        return false;
    }
    const int n = static_cast<int>(cluster_indices.size());
    out->parent.assign(static_cast<std::size_t>(n), -1);
    out->depth.assign(static_cast<std::size_t>(n), 0);
    out->order.resize(static_cast<std::size_t>(n));
    std::iota(out->order.begin(), out->order.end(), 0);
    out->cycle_cuts = 0;
    out->depth_cuts = 0;
    if (n <= 1) return true;
    if (X.rows <= 0) {
        if (error) *error = "BuildReferenceForest: empty vector dimension.";
        return false;
    }
    for (int g : cluster_indices) {
        if (g < 0 || g >= X.cols) {
            if (error) *error = "BuildReferenceForest: cluster index out of range.";
            return false;
        }
    }
    if (policy == "all_roots") return true;

    if (policy == "nn_forest") {
        ClusterHnswIndex index;
        index.Build(X, cluster_indices, hnsw_cfg, std::max(2, std::min(n, 64)));
        index.SetEf(std::max(2, std::min(n, 64)));
        std::vector<int> labels;
        for (int local = 0; local < n; ++local) {
            index.SearchLabelsInto(X.Col(cluster_indices[static_cast<std::size_t>(local)]),
                                   std::min(n, 2), &labels);
            int parent = -1;
            for (int candidate : labels) {
                if (candidate != local) {
                    parent = candidate;
                    break;
                }
            }
            if (parent < 0) {
                // HNSW should return self plus one neighbor for an indexed query.
                // A deterministic exact fallback preserves total-function behavior.
                float best = std::numeric_limits<float>::infinity();
                const float* x = X.Col(cluster_indices[static_cast<std::size_t>(local)]);
                for (int j = 0; j < n; ++j) {
                    if (j == local) continue;
                    const float* y = X.Col(cluster_indices[static_cast<std::size_t>(j)]);
                    float dist2 = 0.0f;
#pragma omp simd reduction(+:dist2)
                    for (int r = 0; r < X.rows; ++r) {
                        const float diff = x[r] - y[r];
                        dist2 += diff * diff;
                    }
                    if (dist2 < best) {
                        best = dist2;
                        parent = j;
                    }
                }
            }
            out->parent[static_cast<std::size_t>(local)] = parent;
        }
    } else if (policy == "random_forest") {
        for (int local = 0; local < n; ++local) {
            const std::uint64_t word = Mix64(seed ^ static_cast<std::uint64_t>(local));
            int parent = static_cast<int>(word % static_cast<std::uint64_t>(n - 1));
            if (parent >= local) ++parent;
            out->parent[static_cast<std::size_t>(local)] = parent;
        }
    } else {
        if (error) *error = "Unsupported linkage reference policy: " + policy;
        return false;
    }

    CutCycles(&out->parent, &out->cycle_cuts);
    BuildDepthLimitedOrder(std::max(1, max_depth), out);
    if (static_cast<int>(out->order.size()) != n) {
        if (error) *error = "BuildReferenceForest: internal forest ordering failure.";
        return false;
    }
    return true;
}

}  // namespace stlq
