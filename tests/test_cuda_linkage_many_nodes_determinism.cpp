#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "stlq/linkage/eval_candidates_cuda.h"
#include "stlq/cuda/cuda_stream_kernels_pool.h"
#include "stlq/quantizer/encoder.h"

namespace {

stlq::CodebookPack MakeRandomCodebooks(int d, const std::vector<int>& h_vec, std::mt19937* rng) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    stlq::CodebookPack pack;
    pack.d = d;
    pack.h_vec = h_vec;
    pack.books.resize(h_vec.size());
    for (std::size_t l = 0; l < h_vec.size(); ++l) {
        const int h = h_vec[l];
        stlq::ColMajorMatrix<float> C(d, h);
        for (int j = 0; j < h; ++j) {
            float norm2 = 0.0f;
            float* col = C.Col(j);
            for (int r = 0; r < d; ++r) {
                const float v = nd(*rng);
                col[r] = v;
                norm2 += v * v;
            }
            const float inv = 1.0f / std::sqrt(std::max(1e-12f, norm2));
            for (int r = 0; r < d; ++r) {
                col[r] *= inv;
            }
        }
        pack.books[l] = std::move(C);
    }
    return pack;
}

void PermuteXBlock(const std::vector<float>& X, int d, int B, const std::vector<int>& perm,
                   std::vector<float>* out) {
    out->assign(static_cast<std::size_t>(d) * static_cast<std::size_t>(B), 0.0f);
    for (int i = 0; i < B; ++i) {
        const int src = perm[static_cast<std::size_t>(i)];
        std::memcpy(out->data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(d),
                    X.data() + static_cast<std::size_t>(src) * static_cast<std::size_t>(d),
                    sizeof(float) * static_cast<std::size_t>(d));
    }
}

struct EvalOut {
    std::vector<int> best_parent;
    std::vector<float> best_cost;
    std::vector<stlq::FullCode> best_B;
    std::vector<float> best_a;
};

}  // namespace

int main() {
#if !defined(STLQ_ENABLE_CUDA) || !STLQ_ENABLE_CUDA
    std::cout << "test_cuda_linkage_many_nodes_determinism: skipped (CUDA disabled)\n";
    return 0;
#else
    const int d = 32;
    const int m = 5;
    const int rfull_cols = 256;
    const int B = 8;
    const int icm_iters = 4;
    const int ils_iters = 20;
    const int perturb_k = 3;
    const std::uint32_t seed = 1234;
    const int forced_root_code = 7;

    std::mt19937 rng(7);
    std::vector<int> h_vec(static_cast<std::size_t>(m), 64);
    stlq::CodebookPack pack = MakeRandomCodebooks(d, h_vec, &rng);
    stlq::Precomp pre;
    if (!stlq::BuildPrecomp(pack, &pre)) {
        std::cerr << "BuildPrecomp failed\n";
        return 2;
    }

    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> R_full(static_cast<std::size_t>(d) * static_cast<std::size_t>(rfull_cols), 0.0f);
    for (float& v : R_full) v = nd(rng);

    std::vector<float> X_block(static_cast<std::size_t>(d) * static_cast<std::size_t>(B), 0.0f);
    for (float& v : X_block) v = nd(rng);

    std::uniform_int_distribution<int> uni_parent(0, rfull_cols - 1);
    std::vector<int> cand_offsets(static_cast<std::size_t>(B + 1), 0);
    std::vector<int> cand_flat;
    std::vector<int> pair_node;
    cand_flat.reserve(1024);
    pair_node.reserve(1024);
    for (int i = 0; i < B; ++i) {
        const int Kp = 3 + (i % 5);
        cand_offsets[static_cast<std::size_t>(i + 1)] = cand_offsets[static_cast<std::size_t>(i)] + Kp;
        for (int t = 0; t < Kp; ++t) {
            cand_flat.push_back(uni_parent(rng));
            pair_node.push_back(i);
        }
    }
    const int Npairs = cand_offsets[static_cast<std::size_t>(B)];

    std::vector<std::uint64_t> node_sample_id_base(static_cast<std::size_t>(B), 0);
    for (int i = 0; i < B; ++i) {
        const std::uint32_t node_local_u32 = 1000u + static_cast<std::uint32_t>(i) * 17u;
        node_sample_id_base[static_cast<std::size_t>(i)] =
            (static_cast<std::uint64_t>(static_cast<std::uint32_t>(forced_root_code)) << 32) |
            static_cast<std::uint64_t>(node_local_u32);
    }

    stlq::CudaPoolConfig pcfg;
    pcfg.device = 0;
    pcfg.allow_tf32 = false;
    stlq::CudaStreamKernelsPool pool(/*num_ctx=*/1, pcfg);
    stlq::CudaCtx* ctx = pool.Acquire();
    if (!ctx) {
        std::cerr << "Acquire CUDA ctx failed\n";
        return 3;
    }

    std::string up_err;
    if (!stlq::CudaLinkageUploadClusterRfull(ctx, R_full.data(), d, rfull_cols, &up_err)) {
        std::cerr << "CudaLinkageUploadClusterRfull failed: " << up_err << "\n";
        return 4;
    }

    auto run = [&](const std::vector<float>& X,
                   const std::vector<int>& cand,
                   const std::vector<int>& pair,
                   const std::vector<int>& offs,
                   const std::vector<std::uint64_t>& base,
                   EvalOut* out) -> bool {
        out->best_parent.assign(static_cast<std::size_t>(B), -1);
        out->best_cost.assign(static_cast<std::size_t>(B), std::numeric_limits<float>::infinity());
        out->best_B.assign(static_cast<std::size_t>(B) * static_cast<std::size_t>(m), 0);
        out->best_a.assign(static_cast<std::size_t>(B) * static_cast<std::size_t>(m), 0.0f);
        std::string err;
        return stlq::EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBase(
            ctx, pre,
            X.data(), d, B,
            cand.data(), pair.data(), offs.data(), Npairs,
            icm_iters, ils_iters, perturb_k, seed,
            base.data(),
            out->best_parent.data(), out->best_cost.data(), out->best_B.data(), out->best_a.data(),
            &err);
    };

    EvalOut out0;
    if (!run(X_block, cand_flat, pair_node, cand_offsets, node_sample_id_base, &out0)) {
        std::cerr << "first run failed\n";
        return 5;
    }

    // Permute node order; outputs should match after un-permuting.
    std::vector<int> perm = {3, 0, 7, 1, 6, 2, 5, 4};
    std::vector<int> inv_perm(static_cast<std::size_t>(B), 0);
    for (int i = 0; i < B; ++i) inv_perm[static_cast<std::size_t>(perm[static_cast<std::size_t>(i)])] = i;

    std::vector<float> Xp;
    PermuteXBlock(X_block, d, B, perm, &Xp);

    std::vector<int> offs_p(static_cast<std::size_t>(B + 1), 0);
    std::vector<int> cand_p;
    std::vector<int> pair_p;
    cand_p.reserve(cand_flat.size());
    pair_p.reserve(pair_node.size());
    for (int i = 0; i < B; ++i) {
        const int src = perm[static_cast<std::size_t>(i)];
        const int start = cand_offsets[static_cast<std::size_t>(src)];
        const int stop = cand_offsets[static_cast<std::size_t>(src + 1)];
        offs_p[static_cast<std::size_t>(i + 1)] = offs_p[static_cast<std::size_t>(i)] + (stop - start);
        for (int j = start; j < stop; ++j) {
            cand_p.push_back(cand_flat[static_cast<std::size_t>(j)]);
            pair_p.push_back(i);
        }
    }

    std::vector<std::uint64_t> base_p(static_cast<std::size_t>(B), 0);
    for (int i = 0; i < B; ++i) {
        base_p[static_cast<std::size_t>(i)] = node_sample_id_base[static_cast<std::size_t>(perm[static_cast<std::size_t>(i)])];
    }

    EvalOut out1;
    if (!run(Xp, cand_p, pair_p, offs_p, base_p, &out1)) {
        std::cerr << "permuted run failed\n";
        return 6;
    }

    int mism = 0;
    for (int orig = 0; orig < B; ++orig) {
        const int ip = inv_perm[static_cast<std::size_t>(orig)];
        if (out0.best_parent[static_cast<std::size_t>(orig)] != out1.best_parent[static_cast<std::size_t>(ip)]) {
            mism++;
        }
        const float c0 = out0.best_cost[static_cast<std::size_t>(orig)];
        const float c1 = out1.best_cost[static_cast<std::size_t>(ip)];
        if (std::fabs(c0 - c1) > 1e-5f) {
            mism++;
        }
        const std::size_t base0 = static_cast<std::size_t>(orig) * static_cast<std::size_t>(m);
        const std::size_t base1 = static_cast<std::size_t>(ip) * static_cast<std::size_t>(m);
        for (int l = 0; l < m; ++l) {
            if (out0.best_B[base0 + static_cast<std::size_t>(l)] != out1.best_B[base1 + static_cast<std::size_t>(l)]) {
                mism++;
                break;
            }
        }
    }

    if (mism != 0) {
        std::cerr << "mismatches=" << mism << "\n";
        return 7;
    }
    std::cout << "test_cuda_linkage_many_nodes_determinism: ok\n";
    return 0;
#endif
}
