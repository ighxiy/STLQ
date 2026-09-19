#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "stlq/common/config.h"
#include "stlq/common/types.h"

namespace stlq {

struct ReferenceForest {
    // Local parent id, or -1 for a root.
    std::vector<int> parent;
    std::vector<int> depth;
    // Parent-before-child order. Roots and siblings use stable local-id order.
    std::vector<int> order;
    int cycle_cuts = 0;
    int depth_cuts = 0;
};

// Construct the policy-controlled preferred-parent forest for one IVF cluster.
// NN references are approximate HNSW nearest neighbors by design. The raw
// functional graph is made deployable by deterministic cycle cuts and max-depth
// cuts; no radial/inward restriction is imposed. The all_roots control returns
// the identity order with no parent edges.
bool BuildReferenceForest(const std::string& policy,
                          const ColMajorMatrix<float>& X,
                          const std::vector<int>& cluster_indices,
                          const HnswConfig& hnsw_cfg,
                          int max_depth,
                          std::uint64_t seed,
                          ReferenceForest* out,
                          std::string* error);

}  // namespace stlq
