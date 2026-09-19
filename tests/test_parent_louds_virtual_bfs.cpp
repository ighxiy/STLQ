#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <vector>

#include "stlq/succinct/parent_louds.h"

namespace {

void Expect(bool condition) {
    if (!condition) std::abort();
}

void TestSequentialDecoders(const std::vector<std::uint32_t>& parent) {
    using stlq::succinct::ParentLOUDS;

    ParentLOUDS louds;
    louds.BuildFromParent1Based(parent.data(), parent.size(),
                               /*select_stride=*/128,
                               /*rank_log2=*/4,
                               /*build_indices=*/true);

    auto sequential = louds.MakeSequentialParentDecoder();
    for (const std::uint32_t expected : parent) {
        Expect(sequential.NextParent1Based() == expected);
    }

    for (const std::size_t start :
         {0u, 1u, 31u, 32u, 63u, 64u, 65u, 127u, 128u, 129u, 2048u}) {
        if (start > parent.size()) continue;
        auto skipped = louds.MakeSequentialParentDecoder();
        skipped.Skip(start);
        for (std::size_t pos = start; pos < parent.size();) {
            Expect(skipped.NextParent1Based() == parent[pos]);
            ++pos;
            const std::size_t count =
                std::min(pos % 67u, parent.size() - pos);
            skipped.Skip(count);
            pos += count;
        }
    }

    const std::size_t root_count = static_cast<std::size_t>(
        std::find_if(parent.begin(), parent.end(),
                     [](std::uint32_t value) { return value != 0; }) -
        parent.begin());
    auto positioned = louds.MakeSequentialParentDecoder();
    positioned.InitializeAfterValidatedRootPrefix(root_count);
    for (std::size_t pos = root_count; pos < parent.size(); ++pos) {
        Expect(positioned.NextParent1Based() == parent[pos]);
    }
}

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
        Expect(a.Parent1BasedOf(i) == parent[i]);
    }

    ParentLOUDS c;
    c.Deserialize(blob_a.data(), blob_a.size(), /*rank_log2=*/4, /*build_indices=*/true);
    for (std::size_t i = 0; i < parent.size(); ++i) {
        Expect(c.Parent1BasedOf(i) == parent[i]);
    }
    TestSequentialDecoders(parent);
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
    Expect(louds.n_nodes() == parent.size());

    for (std::size_t i = 0; i < parent.size(); ++i) {
        Expect(louds.Parent1BasedOf(i) == parent[i]);
    }

    const std::vector<std::uint8_t> blob = louds.Serialize();
    ParentLOUDS loaded;
    loaded.Deserialize(blob.data(), blob.size(), /*rank_log2=*/4, /*build_indices=*/true);
    for (std::size_t i = 0; i < parent.size(); ++i) {
        Expect(loaded.Parent1BasedOf(i) == parent[i]);
    }
    TestSequentialDecoders(parent);
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
        Expect(louds.Parent1BasedOf(i) == parent[i]);
    }
    TestSequentialDecoders(parent);
}

void TestSequentialBoundaryShapes() {
    TestSequentialDecoders({});
    for (const std::uint32_t count :
         {1u, 31u, 32u, 33u, 63u, 64u, 65u,
          127u, 128u, 129u, 4097u}) {
        TestSequentialDecoders(std::vector<std::uint32_t>(count, 0u));

        std::vector<std::uint32_t> chain(count);
        std::vector<std::uint32_t> binary(count);
        std::vector<std::uint32_t> star(count, 1u);
        if (!star.empty()) star[0] = 0u;
        for (std::uint32_t row = 0; row < count; ++row) {
            chain[row] = row;
            binary[row] = (row + 1u) / 2u;
        }
        TestSequentialDecoders(chain);
        TestSequentialDecoders(binary);
        TestSequentialDecoders(star);
    }

    // Complete zero words between non-aligned runs of one bits.
    std::vector<std::uint32_t> sparse(130u, 0u);
    sparse.insert(sparse.end(), 257u, 130u);
    sparse.insert(sparse.end(), 129u, 387u);
    TestSequentialDecoders(sparse);

    std::uint32_t random = 123456789u;
    for (std::uint32_t trial = 0; trial < 32u; ++trial) {
        std::vector<std::uint32_t> parent(1025u);
        std::uint32_t previous = 0;
        for (std::uint32_t row = 1; row < parent.size(); ++row) {
            random = random * 1664525u + 1013904223u;
            previous += random % (row - previous + 1u);
            parent[row] = previous;
        }
        TestSequentialDecoders(parent);
    }
}

}  // namespace

int main() {
    TestRealOnlyRegression();
    TestVirtualFrontBfsParents();
    TestMixedRootsVirtualFrontBfs();
    TestSequentialBoundaryShapes();
    return 0;
}
