#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "stlq/linkage/reference_forest.h"
#include "stlq/linkage/linkage_builder.h"
#include "stlq/linkage/virtual_augment.h"
#include "stlq/pipeline/large_store_hash.h"

namespace {

bool ValidateForest(const stlq::ReferenceForest& forest, int max_depth) {
    const int n = static_cast<int>(forest.parent.size());
    if (static_cast<int>(forest.depth.size()) != n ||
        static_cast<int>(forest.order.size()) != n) return false;
    std::vector<int> position(static_cast<std::size_t>(n), -1);
    for (int i = 0; i < n; ++i) {
        const int node = forest.order[static_cast<std::size_t>(i)];
        if (node < 0 || node >= n || position[static_cast<std::size_t>(node)] >= 0) return false;
        position[static_cast<std::size_t>(node)] = i;
    }
    for (int child = 0; child < n; ++child) {
        const int parent = forest.parent[static_cast<std::size_t>(child)];
        const int depth = forest.depth[static_cast<std::size_t>(child)];
        if (depth < 0 || depth > max_depth || parent == child || parent >= n) return false;
        if (parent < 0) {
            if (depth != 0) return false;
        } else {
            if (position[static_cast<std::size_t>(parent)] >=
                position[static_cast<std::size_t>(child)]) return false;
            if (depth != forest.depth[static_cast<std::size_t>(parent)] + 1) return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    stlq::Config parsed = stlq::DefaultConfig(false);
    std::string error;
    if (!stlq::ApplyOverride("train.linkage.reference_policy", "random_forest", &parsed, &error) ||
        !stlq::ApplyOverride("base.linkage.reference_policy", "causal_nn", &parsed, &error) ||
        !stlq::ApplyOverride("virtual.anchor_policy", "subkmeans", &parsed, &error) ||
        !stlq::ApplyOverride("virtual.subkmeans_iters", "9", &parsed, &error) ||
        parsed.train.linkage.reference_policy != "random_forest" ||
        parsed.base.linkage.reference_policy != "causal_nn" ||
        parsed.virtual_cfg.anchor_policy != "subkmeans" ||
        parsed.virtual_cfg.subkmeans_iters != 9) {
        std::cerr << "new configuration fields did not parse: " << error << "\n";
        return 1;
    }

    stlq::TrainResult identity_train;
    identity_train.R = stlq::ColMajorMatrix<float>(1, 1);
    identity_train.R(0, 0) = 1.0f;
    stlq::Config identity_default = stlq::DefaultConfig(false);
    stlq::Config identity_nn = identity_default;
    identity_nn.base.linkage.reference_policy = "nn_forest";
    stlq::Config identity_causal = identity_default;
    identity_causal.base.linkage.reference_policy = "causal_random";
    stlq::Config identity_subkmeans = identity_default;
    identity_subkmeans.virtual_cfg.enabled = true;
    identity_subkmeans.virtual_cfg.anchor_policy = "subkmeans";
    stlq::Config identity_inactive_subkmeans = identity_default;
    identity_inactive_subkmeans.virtual_cfg.anchor_policy = "subkmeans";
    const std::uint64_t default_hash = stlq::app::ComputeLinkageListStoreHash(
        identity_default, identity_train, 17, 2);
    if (default_hash == stlq::app::ComputeLinkageListStoreHash(identity_nn, identity_train, 17, 2) ||
        default_hash == stlq::app::ComputeLinkageListStoreHash(identity_causal, identity_train, 17, 2) ||
        default_hash == stlq::app::ComputeLinkageListStoreHash(identity_subkmeans, identity_train, 17, 2) ||
        default_hash != stlq::app::ComputeLinkageListStoreHash(
                            identity_inactive_subkmeans, identity_train, 17, 2)) {
        std::cerr << "artifact identity does not bind the new active policies\n";
        return 2;
    }

    stlq::ColMajorMatrix<float> x(2, 9);
    for (int i = 0; i < x.cols; ++i) {
        x(0, i) = static_cast<float>(i);
        x(1, i) = static_cast<float>((i * i) % 7);
    }
    std::vector<int> ids(static_cast<std::size_t>(x.cols));
    for (int i = 0; i < x.cols; ++i) ids[static_cast<std::size_t>(i)] = i;
    stlq::HnswConfig hnsw;
    hnsw.M = 8;

    stlq::ReferenceForest random_a;
    stlq::ReferenceForest random_b;
    if (!stlq::BuildReferenceForest("random_forest", x, ids, hnsw, 2, 1234,
                                    &random_a, &error) ||
        !stlq::BuildReferenceForest("random_forest", x, ids, hnsw, 2, 1234,
                                    &random_b, &error)) {
        std::cerr << error << "\n";
        return 3;
    }
    if (random_a.parent != random_b.parent || random_a.depth != random_b.depth ||
        random_a.order != random_b.order || !ValidateForest(random_a, 2)) {
        std::cerr << "random reference forest is invalid or nondeterministic\n";
        return 4;
    }

    stlq::ReferenceForest nearest;
    if (!stlq::BuildReferenceForest("nn_forest", x, ids, hnsw, 3, 9,
                                    &nearest, &error) ||
        !ValidateForest(nearest, 3)) {
        std::cerr << "NN reference forest failed: " << error << "\n";
        return 5;
    }

    stlq::ReferenceForest all_roots;
    if (!stlq::BuildReferenceForest("all_roots", x, ids, hnsw, 3, 9,
                                    &all_roots, &error) ||
        !ValidateForest(all_roots, 3) ||
        std::any_of(all_roots.parent.begin(), all_roots.parent.end(),
                    [](int parent) { return parent >= 0; }) ||
        all_roots.order != ids) {
        std::cerr << "all-roots reference control failed: " << error << "\n";
        return 6;
    }

    stlq::ColMajorMatrix<float> values(1, 6);
    values(0, 0) = 0.0f;
    values(0, 1) = 2.0f;
    values(0, 2) = 4.0f;
    values(0, 3) = 100.0f;
    values(0, 4) = 102.0f;
    values(0, 5) = 104.0f;
    std::vector<int> value_ids{0, 1, 2, 3, 4, 5};
    stlq::VirtualConfig virtual_cfg;
    virtual_cfg.anchor_policy = "subkmeans";
    virtual_cfg.subkmeans_iters = 8;
    stlq::ColMajorMatrix<float> centers;
    std::vector<int> roots;
    const std::vector<std::uint32_t> no_knn_ids;
    const std::vector<float> no_knn_dists;
    if (!stlq::BuildVirtualAnchorCentersFromKnnTables(
            virtual_cfg, values, value_ids, no_knn_ids, no_knn_dists,
            0, 0, 2, 7, 42, &centers, &roots, &error)) {
        std::cerr << "classic sub-kmeans failed: " << error << "\n";
        return 7;
    }
    if (centers.rows != 1 || centers.cols != 2 || roots != std::vector<int>({7, 7})) {
        std::cerr << "classic sub-kmeans output shape mismatch\n";
        return 8;
    }
    std::vector<float> means{centers(0, 0), centers(0, 1)};
    std::sort(means.begin(), means.end());
    if (std::abs(means[0] - 2.0f) > 1e-6f ||
        std::abs(means[1] - 102.0f) > 1e-6f) {
        std::cerr << "sub-kmeans did not produce arithmetic Euclidean centroids: "
                  << means[0] << ", " << means[1] << "\n";
        return 9;
    }

    // Exercise the public non-streaming producer, including strict-improvement
    // encoding and depth-order finalization, for all experimental policies.
    stlq::CodebookPack root_pack;
    stlq::CodebookPack edge_pack;
    root_pack.d = 2;
    edge_pack.d = 2;
    root_pack.h_vec = {2, 3};
    edge_pack.h_vec = {3, 3};
    root_pack.books = {stlq::ColMajorMatrix<float>(2, 2),
                       stlq::ColMajorMatrix<float>(2, 3)};
    edge_pack.books = {stlq::ColMajorMatrix<float>(2, 3),
                       stlq::ColMajorMatrix<float>(2, 3)};
    root_pack.books[0](0, 0) = 1.0f;
    root_pack.books[0](1, 1) = 1.0f;
    root_pack.books[1](1, 0) = 1.0f;
    root_pack.books[1](0, 1) = 1.0f;
    root_pack.books[1](0, 2) = 1.0f;
    root_pack.books[1](1, 2) = 1.0f;
    edge_pack.books[0](0, 0) = 1.0f;
    edge_pack.books[0](1, 1) = 1.0f;
    edge_pack.books[0](0, 2) = 1.0f;
    edge_pack.books[0](1, 2) = 1.0f;
    edge_pack.books[1](1, 0) = 1.0f;
    edge_pack.books[1](0, 1) = 1.0f;
    edge_pack.books[1](0, 2) = 1.0f;
    edge_pack.books[1](1, 2) = -1.0f;

    stlq::ColMajorMatrix<float> samples(2, 6);
    const float sample_values[12] = {
        1.0f, 0.0f, 1.0f, 1.0f, 2.0f, 1.0f,
        2.0f, 2.0f, 3.0f, 2.0f, 3.0f, 3.0f};
    for (int i = 0; i < 6; ++i) {
        samples(0, i) = sample_values[2 * i];
        samples(1, i) = sample_values[2 * i + 1];
    }
    for (const std::string policy : {std::string("nn_forest"),
                                     std::string("random_forest"),
                                     std::string("all_roots"),
                                     std::string("causal_nn"),
                                     std::string("causal_random")}) {
        stlq::BaseEncoding base;
        base.B = stlq::ColMajorMatrix<stlq::FullCode>(2, 6);
        base.a = stlq::ColMajorMatrix<float>(2, 6);
        for (int i = 0; i < 6; ++i) {
            base.B(0, i) = 0;
            base.B(1, i) = 0;
            base.a(0, i) = 1.0f;
            base.a(1, i) = 0.0f;
        }
        stlq::LinkageBuildConfig linkage_cfg;
        linkage_cfg.reference_policy = policy;
        linkage_cfg.max_depth = 3;
        linkage_cfg.num_layers = 4;
        linkage_cfg.root_percentile = 0.2;
        linkage_cfg.icm_round = 0;
        linkage_cfg.use_ils = false;
        stlq::LinkageStructure linkage;
        stlq::ColMajorMatrix<float> recon;
        if (!stlq::BuildLinkageTwoCodebook(linkage_cfg, hnsw, samples, &base,
                                           root_pack, edge_pack, &linkage, &recon,
                                           &error)) {
            std::cerr << "public reference-forest linkage failed: " << error << "\n";
            return 10;
        }
        if (linkage.clusters.size() != 1 || recon.rows != 2 || recon.cols != 6 ||
            linkage.max_depth > linkage_cfg.max_depth) {
            std::cerr << "public reference-forest linkage output mismatch: clusters="
                      << linkage.clusters.size() << " recon=" << recon.rows << "x" << recon.cols
                      << " depth=" << linkage.max_depth << "\n";
            return 11;
        }
    }

    std::cout << "reference forest and classic sub-kmeans checks passed\n";
    return 0;
}
