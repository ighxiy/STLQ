#include <cassert>
#include <cstdint>
#include <vector>

#include "stlq/succinct/parent_louds.h"

namespace {

void TestRealOnlyRegression() {
    using stlq::succinct::ParentLOUDS;

    // Real-only, BFS order for the augmented tree with an implicit super-root:
    // roots first, then level-order within the forest.
    //
    // Roots: 0, 1
    // 0 -> {2,3}
    // 2 -> {6}
    // 1 -> {4,5}
    // 5 -> {7}
    const std::vector<std::uint32_t> parent = {
        0,  // 0 root
        0,  // 1 root
        1,  // 2 child of 0
        1,  // 3 child of 0
        2,  // 4 child of 1
        2,  // 5 child of 1
        3,  // 6 child of 2
        6,  // 7 child of 5
    };

    ParentLOUDS a;
    a.BuildFromParent1Based(parent.data(), parent.size(), /*select_stride=*/128, /*rank_log2=*/4, /*build_indices=*/true);
    const std::vector<std::uint8_t> blob_a = a.Serialize();
    for (std::size_t i = 0; i < parent.size(); ++i) {
        assert(a.Parent1BasedOf(i) == parent[i]);
    }

    ParentLOUDS c;
    c.Deserialize(blob_a.data(), blob_a.size(), /*rank_log2=*/4, /*build_indices=*/true);
    for (std::size_t i = 0; i < parent.size(); ++i) {
        assert(c.Parent1BasedOf(i) == parent[i]);
    }
}

void TestVirtualFrontBfsParents() {
    using stlq::succinct::ParentLOUDS;

    // virtual-front, full-node BFS order:
    // roots: 0(v), 1(v)
    // 0 -> {2,3}
    // 1 -> {4}
    // 2 -> {5}
    // 4 -> {6,7}
    const std::vector<std::uint32_t> parent = {0, 0, 1, 1, 2, 3, 5, 5};

    ParentLOUDS louds;
    louds.BuildFromParent1Based(parent.data(), parent.size(), /*select_stride=*/128, /*rank_log2=*/4, /*build_indices=*/true);
    assert(louds.n_nodes() == parent.size());

    for (std::size_t i = 0; i < parent.size(); ++i) {
        assert(louds.Parent1BasedOf(i) == parent[i]);
    }

    const std::vector<std::uint8_t> blob = louds.Serialize();
    ParentLOUDS loaded;
    loaded.Deserialize(blob.data(), blob.size(), /*rank_log2=*/4, /*build_indices=*/true);
    for (std::size_t i = 0; i < parent.size(); ++i) {
        assert(loaded.Parent1BasedOf(i) == parent[i]);
    }
}

void TestMixedRootsVirtualFrontBfs() {
    using stlq::succinct::ParentLOUDS;

    // roots: 0(v), 1(v), 2(v), 3(r), 4(r)
    // 0 -> {5}
    // 3 -> {6}
    // 4 -> {7}
    const std::vector<std::uint32_t> parent = {0, 0, 0, 0, 0, 1, 4, 5};

    ParentLOUDS louds;
    louds.BuildFromParent1Based(parent.data(), parent.size(), /*select_stride=*/128, /*rank_log2=*/4, /*build_indices=*/true);
    for (std::size_t i = 0; i < parent.size(); ++i) {
        assert(louds.Parent1BasedOf(i) == parent[i]);
    }
}

}  // namespace

int main() {
    TestRealOnlyRegression();
    TestVirtualFrontBfsParents();
    TestMixedRootsVirtualFrontBfs();
    return 0;
}
