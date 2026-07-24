#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "stlq/common/types.h"

namespace stlq {

namespace {

inline std::uint64_t StableKeyForLocal(const LinkageStructure::Cluster& c, int local) {
    const int n_virt = c.n_virtual;
    if (local < n_virt) {
        // virtual: stable by local id
        return static_cast<std::uint64_t>(local);
    }
    const int real_idx = local - n_virt;
    if (real_idx < 0 || real_idx >= static_cast<int>(c.indices.size())) {
        return static_cast<std::uint64_t>(local);
    }
    // real: stable by global id (usually unique and deterministic)
    return static_cast<std::uint64_t>(static_cast<std::uint32_t>(c.indices[static_cast<std::size_t>(real_idx)]));
}

inline void FinalizeLinkageClusterBfsVirtualFrontInPlace(LinkageStructure::Cluster* cluster) {
    if (!cluster) {
        return;
    }
    const int n_virt = std::max(0, cluster->n_virtual);
    const int n_real = std::max(0, cluster->n_real);
    const int n_total = n_virt + n_real;

    if (n_total <= 0) {
        cluster->depth_offsets.clear();
        cluster->n_real = 0;
        cluster->n_virtual = 0;
        cluster->indices.clear();
        cluster->parent_local.clear();
        return;
    }

    if (static_cast<int>(cluster->indices.size()) != n_real) {
        // Best-effort: resize to match declared n_real (callers are expected to set this correctly).
        cluster->indices.resize(static_cast<std::size_t>(n_real), 0);
    }
    if (static_cast<int>(cluster->parent_local.size()) != n_total) {
        cluster->parent_local.resize(static_cast<std::size_t>(n_total), 0);
    }

#ifndef NDEBUG
    // Virtual nodes must be roots in this scheme.
    for (int v = 0; v < n_virt; ++v) {
        if (cluster->parent_local[static_cast<std::size_t>(v)] != 0) {
            throw std::runtime_error("FinalizeLinkageCluster: virtual node must be a root (parent==0)");
        }
    }
#endif

    // Build adjacency in old local id space.
    std::vector<std::vector<int>> children(static_cast<std::size_t>(n_total));
    std::vector<int> roots;
    roots.reserve(static_cast<std::size_t>(n_total));
    for (int u = 0; u < n_total; ++u) {
        const int p1 = cluster->parent_local[static_cast<std::size_t>(u)];
        if (p1 == 0) {
            roots.push_back(u);
            continue;
        }
        const int p = p1 - 1;
        if (p >= 0 && p < n_total) {
            children[static_cast<std::size_t>(p)].push_back(u);
        } else {
            // Invalid parent: treat as root.
            roots.push_back(u);
        }
    }

    // Sort children lists: virtual-first, then real; stable within each by key.
    auto child_less = [&](int a, int b) {
        const bool va = a < n_virt;
        const bool vb = b < n_virt;
        if (va != vb) {
            return va > vb;  // virtual first
        }
        const std::uint64_t ka = StableKeyForLocal(*cluster, a);
        const std::uint64_t kb = StableKeyForLocal(*cluster, b);
        if (ka != kb) {
            return ka < kb;
        }
        return a < b;
    };
    for (auto& kids : children) {
        if (kids.size() > 1) {
            std::stable_sort(kids.begin(), kids.end(), child_less);
        }
    }

    // Sort roots: virtual roots first, then real roots.
    if (roots.size() > 1) {
        std::stable_sort(roots.begin(), roots.end(), child_less);
    }

    // BFS over all nodes (forest). Compute order and depths in old local space.
    std::vector<int> bfs_old;
    bfs_old.reserve(static_cast<std::size_t>(n_total));
    std::vector<int> depth_old(static_cast<std::size_t>(n_total), 0);

    std::vector<int> cur = roots;
    std::vector<int> next;
    next.reserve(static_cast<std::size_t>(n_total));
    int dep = 0;
    std::vector<char> seen(static_cast<std::size_t>(n_total), 0);
    while (!cur.empty()) {
        next.clear();
        for (int u : cur) {
            if (u < 0 || u >= n_total) {
                continue;
            }
            if (seen[static_cast<std::size_t>(u)] != 0) {
                continue;
            }
            seen[static_cast<std::size_t>(u)] = 1;
            bfs_old.push_back(u);
            depth_old[static_cast<std::size_t>(u)] = dep;
        }
        for (int u : cur) {
            if (u < 0 || u >= n_total) {
                continue;
            }
            const auto& kids = children[static_cast<std::size_t>(u)];
            next.insert(next.end(), kids.begin(), kids.end());
        }
        cur.swap(next);
        ++dep;
    }

    // If cycles/disconnected nodes exist, append remaining nodes deterministically as extra roots.
    if (static_cast<int>(bfs_old.size()) != n_total) {
        std::vector<int> missing;
        missing.reserve(static_cast<std::size_t>(n_total - static_cast<int>(bfs_old.size())));
        for (int u = 0; u < n_total; ++u) {
            if (seen[static_cast<std::size_t>(u)] == 0) {
                missing.push_back(u);
            }
        }
        if (!missing.empty()) {
            std::stable_sort(missing.begin(), missing.end(), child_less);
            for (int u : missing) {
                bfs_old.push_back(u);
                depth_old[static_cast<std::size_t>(u)] = 0;
            }
        }
    }

    // Enforce that virtual nodes occupy the prefix in the final layout.
    for (int i = 0; i < n_virt && i < static_cast<int>(bfs_old.size()); ++i) {
        if (bfs_old[static_cast<std::size_t>(i)] >= n_virt) {
#ifndef NDEBUG
            throw std::runtime_error("FinalizeLinkageCluster: expected all virtual nodes to appear first in BFS order");
#else
            break;
#endif
        }
    }

    // old_local -> new_local
    std::vector<int> old2new(static_cast<std::size_t>(n_total), -1);
    for (int new_local = 0; new_local < n_total; ++new_local) {
        const int old_local = bfs_old[static_cast<std::size_t>(new_local)];
        if (old_local >= 0 && old_local < n_total) {
            old2new[static_cast<std::size_t>(old_local)] = new_local;
        }
    }

    // Rebuild parent_local in new local space.
    std::vector<int> parent_new(static_cast<std::size_t>(n_total), 0);
    for (int new_local = 0; new_local < n_total; ++new_local) {
        const int old_local = bfs_old[static_cast<std::size_t>(new_local)];
        const int p1 = cluster->parent_local[static_cast<std::size_t>(old_local)];
        if (p1 == 0) {
            parent_new[static_cast<std::size_t>(new_local)] = 0;
            continue;
        }
        const int old_p = p1 - 1;
        if (old_p < 0 || old_p >= n_total) {
            parent_new[static_cast<std::size_t>(new_local)] = 0;
            continue;
        }
        const int new_p = old2new[static_cast<std::size_t>(old_p)];
        parent_new[static_cast<std::size_t>(new_local)] = (new_p >= 0) ? (new_p + 1) : 0;
    }

    // Rebuild indices (real-only) in new real_idx order.
    std::vector<int> indices_new(static_cast<std::size_t>(n_real), 0);
    for (int new_local = n_virt; new_local < n_total; ++new_local) {
        const int new_real_idx = new_local - n_virt;
        const int old_local = bfs_old[static_cast<std::size_t>(new_local)];
        const int old_real_idx = old_local - n_virt;
        if (old_real_idx >= 0 && old_real_idx < n_real) {
            indices_new[static_cast<std::size_t>(new_real_idx)] =
                cluster->indices[static_cast<std::size_t>(old_real_idx)];
        }
    }

    // Build depth_offsets over real_idx order.
    if (n_real <= 0) {
        cluster->depth_offsets.assign(2, 0);
    } else {
        int max_depth_real = 0;
        for (int new_local = n_virt; new_local < n_total; ++new_local) {
            const int old_local = bfs_old[static_cast<std::size_t>(new_local)];
            max_depth_real = std::max(max_depth_real, depth_old[static_cast<std::size_t>(old_local)]);
        }
        std::vector<int> offsets(static_cast<std::size_t>(max_depth_real + 2), 0);
        for (int new_local = n_virt; new_local < n_total; ++new_local) {
            const int old_local = bfs_old[static_cast<std::size_t>(new_local)];
            const int d = std::max(0, depth_old[static_cast<std::size_t>(old_local)]);
            offsets[static_cast<std::size_t>(d + 1)] += 1;
        }
        for (int d = 1; d <= max_depth_real + 1; ++d) {
            offsets[static_cast<std::size_t>(d)] += offsets[static_cast<std::size_t>(d - 1)];
        }
        cluster->depth_offsets.swap(offsets);
    }

    cluster->indices.swap(indices_new);
    cluster->parent_local.swap(parent_new);

#ifndef NDEBUG
    // Basic invariants for LOUDS v1 compatibility:
    // parent must precede child in local BFS order.
    for (int local = 0; local < n_total; ++local) {
        const int p1 = cluster->parent_local[static_cast<std::size_t>(local)];
        if (p1 == 0) continue;
        const int p = p1 - 1;
        if (!(p >= 0 && p < local)) {
            throw std::runtime_error("FinalizeLinkageCluster: parent must precede child in BFS order");
        }
    }
#endif
}

}  // namespace

// Same as FinalizeLinkageClusterDepthOrderInPlace, but accepts a precomputed depth array.
// After the migration, this function performs BFS/level-order finalize (virtual-front) and does not
// rely on `depth_real` (kept for API compatibility with existing builders).
inline void FinalizeLinkageClusterDepthOrderInPlaceFromDepth(LinkageStructure::Cluster* cluster,
                                                           const int* /*depth_real*/) {
    FinalizeLinkageClusterBfsVirtualFrontInPlace(cluster);
}

// Finalize cluster storage:
// - Convert to virtual-front + full-node BFS/level-order local layout (LOUDS v1 compatible).
// - Recompute depth_offsets over REAL nodes (real_idx space).
inline void FinalizeLinkageClusterDepthOrderInPlace(LinkageStructure::Cluster* cluster) {
    FinalizeLinkageClusterBfsVirtualFrontInPlace(cluster);
}

}  // namespace stlq
