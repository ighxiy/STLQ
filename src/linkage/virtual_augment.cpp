#include "stlq/linkage/virtual_augment.h"

#include <atomic>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

#include <omp.h>

#include "stlq/core/blas.h"
#include "stlq/linkage/linkage_build_profile.h"
#include "stlq/knn/hnsw_cluster_knn.h"
#include "stlq/common/logger.h"
#include "stlq/quantizer/cost_utils.h"
#include "stlq/quantizer/icm.h"
#include "stlq/quantizer/linear_algebra.h"
#if defined(STLQ_ENABLE_CUDA)
#include "stlq/linkage/linkage_encode_cuda.h"
#include "stlq/cuda/cuda_stream_kernels_pool.h"
#endif

namespace stlq
{
    namespace
    {
        constexpr float kUmapEps = 1e-6f;

        struct UmapScratch
        {
            std::vector<const float*> Xp;
            std::vector<float> rho;
            std::vector<float> sigma;
            std::vector<int> nbrs_flat;
            std::vector<float> pdir_flat;
            std::vector<float> wsym_flat;
            std::vector<float> deg;

            std::vector<int> cand;
            std::vector<int> cand2;
            std::vector<int> peaks;
            std::vector<int> seeds;
            std::vector<std::uint8_t> in_seed;

            std::vector<std::pair<float, int>> wj;
        };

        void PrintAugmentProgress(int done, int total) {
            // Called from an OpenMP parallel region; avoid locks by letting only one
            // thread print each `done` value.
            static std::atomic<int> last_printed{-1};
            int prev = last_printed.load(std::memory_order_relaxed);
            while (done > prev &&
                !last_printed.compare_exchange_weak(prev, done,
                                                    std::memory_order_relaxed,
                                                    std::memory_order_relaxed)) {
            }
            if (done <= prev) {
                return;
            }
            std::fprintf(stderr, "\rprocessed %d/%d   ", done, total);
            if (done >= total) {
                std::fprintf(stderr, "\n");
            }
            std::fflush(stderr);
        }

        inline float WSym(float pij, float pji) {
            return pij + pji - pij * pji;
        }

        inline float PIj(float d, float rho, float sigma) {
            const float v = d - rho;
            if (v <= 0.0f) {
                return 1.0f;
            }
            return std::exp(-v / sigma);
        }

        inline float SquaredL2(const float* a, const float* b, int d) {
            float sum = 0.0f;
#pragma omp simd reduction(+:sum)
            for (int r = 0; r < d; ++r) {
                const float diff = a[r] - b[r];
                sum += diff * diff;
            }
            return sum;
        }

        bool BuildUmapCentersForClusterFromKnnTables(const VirtualConfig& vcfg,
                                                     const ColMajorMatrix<float>& X,
                                                     const std::vector<int>& cluster_cols,
                                                     const std::vector<std::uint32_t>& knn_ids_flat,
                                                     const std::vector<float>& knn_dists_flat,
                                                     int k_graph_total,
                                                     int k_graph_umap,
                                                     int k_virtual,
                                                     int forced_root_code,
                                                     ColMajorMatrix<float>* centers,
                                                     std::vector<int>* forced_codes,
                                                     std::string* error);

        // Returns pdir for directed edges i->neighbor (neighbors exclude self, length k_graph).
        [[maybe_unused]] void UmapRhoSigma(const float* dists_flat,
                                           int n_local,
                                           int k_graph,
                                           int local_connectivity,
                                           float target,
                                           std::vector<float>* rho,
                                           std::vector<float>* sigma) {
            rho->resize(static_cast<std::size_t>(n_local));
            sigma->resize(static_cast<std::size_t>(n_local));
            std::fill(rho->begin(), rho->end(), 0.0f);
            std::fill(sigma->begin(), sigma->end(), 1.0f);

            const float eps = kUmapEps;

            for (int i = 0; i < n_local; ++i) {
                const float* drow = dists_flat + static_cast<std::size_t>(i) * k_graph;
                const int k_eff = k_graph;
                if (k_eff <= 0) {
                    (*rho)[static_cast<std::size_t>(i)] = 0.0f;
                    (*sigma)[static_cast<std::size_t>(i)] = 1.0f;
                    continue;
                }

                const int lc = std::min(std::max(1, local_connectivity), k_eff);
                const float rho_i = std::max(drow[lc - 1], 0.0f);
                (*rho)[static_cast<std::size_t>(i)] = rho_i;

                const float d_max = std::max(drow[k_eff - 1], eps);
                float lo = eps;
                float hi = d_max;

                int ones_cnt = 0;
                for (int t = 0; t < k_eff; ++t) {
                    if (drow[t] <= rho_i + eps) {
                        ones_cnt += 1;
                    }
                }
                if (target <= static_cast<float>(ones_cnt) + eps) {
                    (*sigma)[static_cast<std::size_t>(i)] = eps;
                    continue;
                }

                for (int it = 0; it < 64; ++it) {
                    const float mid = 0.5f * (lo + hi);
                    float s = 0.0f;
                    for (int t = 0; t < k_eff; ++t) {
                        const float v = drow[t] - rho_i;
                        s += (v <= 0.0f) ? 1.0f : std::exp(-v / mid);
                    }
                    if (s > target) {
                        hi = mid;
                    }
                    else {
                        lo = mid;
                    }
                }
                (*sigma)[static_cast<std::size_t>(i)] = hi;
            }
        }

        void UmapRhoSigmaStrided(const float* dists_flat,
                                 int stride,
                                 int n_local,
                                 int k_graph,
                                 int local_connectivity,
                                 float target,
                                 std::vector<float>* rho,
                                 std::vector<float>* sigma) {
            rho->resize(static_cast<std::size_t>(n_local));
            sigma->resize(static_cast<std::size_t>(n_local));
            std::fill(rho->begin(), rho->end(), 0.0f);
            std::fill(sigma->begin(), sigma->end(), 1.0f);

            const float eps = kUmapEps;
            stride = std::max(stride, k_graph);

            for (int i = 0; i < n_local; ++i) {
                const float* drow = dists_flat + static_cast<std::size_t>(i) * stride;
                const int k_eff = k_graph;
                if (k_eff <= 0) {
                    (*rho)[static_cast<std::size_t>(i)] = 0.0f;
                    (*sigma)[static_cast<std::size_t>(i)] = 1.0f;
                    continue;
                }

                const int lc = std::min(std::max(1, local_connectivity), k_eff);
                const float rho_i = std::max(drow[lc - 1], 0.0f);
                (*rho)[static_cast<std::size_t>(i)] = rho_i;

                const float d_max = std::max(drow[k_eff - 1], eps);
                float lo = eps;
                float hi = d_max;

                int ones_cnt = 0;
                for (int t = 0; t < k_eff; ++t) {
                    if (drow[t] <= rho_i + eps) {
                        ones_cnt += 1;
                    }
                }
                if (target <= static_cast<float>(ones_cnt) + eps) {
                    (*sigma)[static_cast<std::size_t>(i)] = eps;
                    continue;
                }

                for (int it = 0; it < 64; ++it) {
                    const float mid = 0.5f * (lo + hi);
                    float s = 0.0f;
                    for (int t = 0; t < k_eff; ++t) {
                        const float v = drow[t] - rho_i;
                        s += (v <= 0.0f) ? 1.0f : std::exp(-v / mid);
                    }
                    if (s > target) {
                        hi = mid;
                    }
                    else {
                        lo = mid;
                    }
                }
                (*sigma)[static_cast<std::size_t>(i)] = hi;
            }
        }

        inline float GetP(const int* nbrs_flat,
                          const float* pdir_flat,
                          int k_graph,
                          int i,
                          int j) {
            const int* nbrs = nbrs_flat + static_cast<std::size_t>(i) * k_graph;
            const float* p = pdir_flat + static_cast<std::size_t>(i) * k_graph;
            for (int t = 0; t < k_graph; ++t) {
                if (nbrs[t] == j) {
                    return p[t];
                }
            }
            return 0.0f;
        }

        inline float GetWSymFromPrecomp(const int* nbrs_flat,
                                        const float* wsym_flat,
                                        int k_graph,
                                        int i,
                                        int j) {
            // Preserve old semantics: w(i,j)=WSym(P(i->j),P(j->i)), even if only one direction exists.
            const int* nbrs_i = nbrs_flat + static_cast<std::size_t>(i) * k_graph;
            const float* w_i = wsym_flat + static_cast<std::size_t>(i) * k_graph;
            for (int t = 0; t < k_graph; ++t) {
                if (nbrs_i[t] == j) {
                    return w_i[t];
                }
            }
            const int* nbrs_j = nbrs_flat + static_cast<std::size_t>(j) * k_graph;
            const float* w_j = wsym_flat + static_cast<std::size_t>(j) * k_graph;
            for (int t = 0; t < k_graph; ++t) {
                if (nbrs_j[t] == i) {
                    return w_j[t];
                }
            }
            return 0.0f;
        }

        [[maybe_unused]] std::vector<int> SelectSeedsUmapSimple(const int* nbrs_flat,
                                                                const float* pdir_flat,
                                                                int n_local,
                                                                int k_graph,
                                                                const std::vector<float>& deg,
                                                                int k_virtual,
                                                                float overlap_thr,
                                                                bool prefer_peaks) {
            k_virtual = std::min(k_virtual, n_local);
            if (k_virtual <= 0) {
                return {};
            }

            std::vector<int> cand;
            cand.reserve(static_cast<std::size_t>(n_local) * 2);

            std::vector<int> cand2(static_cast<std::size_t>(n_local), 0);
            std::iota(cand2.begin(), cand2.end(), 0);
            std::sort(cand2.begin(), cand2.end(),
                      [&](int a, int b)
                      {
                          return deg[static_cast<std::size_t>(a)] > deg[static_cast<std::size_t>(b)];
                      });

            if (prefer_peaks) {
                std::vector<int> peaks;
                peaks.reserve(static_cast<std::size_t>(n_local));
                for (int i = 0; i < n_local; ++i) {
                    const float di = deg[static_cast<std::size_t>(i)];
                    bool peak = true;
                    const int* nbrs = nbrs_flat + static_cast<std::size_t>(i) * k_graph;
                    for (int t = 0; t < k_graph; ++t) {
                        const int j = nbrs[t];
                        if (deg[static_cast<std::size_t>(j)] > di) {
                            peak = false;
                            break;
                        }
                    }
                    if (peak) {
                        peaks.push_back(i);
                    }
                }
                std::sort(peaks.begin(), peaks.end(),
                          [&](int a, int b)
                          {
                              return deg[static_cast<std::size_t>(a)] > deg[static_cast<std::size_t>(b)];
                          });
                cand.insert(cand.end(), peaks.begin(), peaks.end());
            }
            cand.insert(cand.end(), cand2.begin(), cand2.end());

            std::vector<int> seeds;
            seeds.reserve(static_cast<std::size_t>(k_virtual));
            std::vector<std::uint8_t> in_seed(static_cast<std::size_t>(n_local), 0);

            for (int i : cand) {
                if (static_cast<int>(seeds.size()) >= k_virtual) {
                    break;
                }
                if (in_seed[static_cast<std::size_t>(i)]) {
                    continue;
                }
                if (seeds.empty()) {
                    seeds.push_back(i);
                    in_seed[static_cast<std::size_t>(i)] = 1;
                    continue;
                }
                float best = 0.0f;
                for (int s : seeds) {
                    const float pij = GetP(nbrs_flat, pdir_flat, k_graph, i, s);
                    const float pji = GetP(nbrs_flat, pdir_flat, k_graph, s, i);
                    const float w = WSym(pij, pji);
                    if (w > best) {
                        best = w;
                    }
                }
                if (best <= overlap_thr) {
                    seeds.push_back(i);
                    in_seed[static_cast<std::size_t>(i)] = 1;
                }
            }

            if (static_cast<int>(seeds.size()) < k_virtual) {
                for (int i : cand2) {
                    if (static_cast<int>(seeds.size()) >= k_virtual) {
                        break;
                    }
                    if (in_seed[static_cast<std::size_t>(i)]) {
                        continue;
                    }
                    seeds.push_back(i);
                    in_seed[static_cast<std::size_t>(i)] = 1;
                }
            }

            return seeds;
        }

        void SelectSeedsUmapSimpleInto(const int* nbrs_flat,
                                       const float* pdir_flat,
                                       const float* wsym_flat,
                                       int n_local,
                                       int k_graph,
                                       const std::vector<float>& deg,
                                       int k_virtual,
                                       float overlap_thr,
                                       bool prefer_peaks,
                                       UmapScratch* scratch,
                                       std::vector<int>* seeds_out) {
            if (scratch == nullptr || seeds_out == nullptr) {
                return;
            }
            k_virtual = std::min(k_virtual, n_local);
            if (k_virtual <= 0) {
                seeds_out->clear();
                return;
            }

            auto& cand = scratch->cand;
            auto& cand2 = scratch->cand2;
            auto& peaks = scratch->peaks;
            auto& in_seed = scratch->in_seed;

            cand.clear();
            cand.reserve(static_cast<std::size_t>(n_local) * 2);

            cand2.resize(static_cast<std::size_t>(n_local));
            std::iota(cand2.begin(), cand2.end(), 0);
            std::sort(cand2.begin(), cand2.end(),
                      [&](int a, int b)
                      {
                          return deg[static_cast<std::size_t>(a)] >
                              deg[static_cast<std::size_t>(b)];
                      });

            if (prefer_peaks) {
                peaks.clear();
                peaks.reserve(static_cast<std::size_t>(n_local));
                for (int i = 0; i < n_local; ++i) {
                    const float di = deg[static_cast<std::size_t>(i)];
                    bool peak = true;
                    const int* nbrs = nbrs_flat + static_cast<std::size_t>(i) * k_graph;
                    for (int t = 0; t < k_graph; ++t) {
                        const int j = nbrs[t];
                        if (deg[static_cast<std::size_t>(j)] > di) {
                            peak = false;
                            break;
                        }
                    }
                    if (peak) {
                        peaks.push_back(i);
                    }
                }
                std::sort(peaks.begin(), peaks.end(),
                          [&](int a, int b)
                          {
                              return deg[static_cast<std::size_t>(a)] >
                                  deg[static_cast<std::size_t>(b)];
                          });
                cand.insert(cand.end(), peaks.begin(), peaks.end());
            }
            cand.insert(cand.end(), cand2.begin(), cand2.end());

            seeds_out->clear();
            seeds_out->reserve(static_cast<std::size_t>(k_virtual));
            in_seed.resize(static_cast<std::size_t>(n_local));
            std::fill(in_seed.begin(), in_seed.end(), 0);

            for (int i : cand) {
                if (static_cast<int>(seeds_out->size()) >= k_virtual) {
                    break;
                }
                if (in_seed[static_cast<std::size_t>(i)]) {
                    continue;
                }
                if (seeds_out->empty()) {
                    seeds_out->push_back(i);
                    in_seed[static_cast<std::size_t>(i)] = 1;
                    continue;
                }
                float best = 0.0f;
                for (int s : *seeds_out) {
                    const float w = (wsym_flat != nullptr)
                                        ? GetWSymFromPrecomp(nbrs_flat, wsym_flat, k_graph, i, s)
                                        : WSym(GetP(nbrs_flat, pdir_flat, k_graph, i, s),
                                               GetP(nbrs_flat, pdir_flat, k_graph, s, i));
                    if (w > best) {
                        best = w;
                    }
                }
                if (best <= overlap_thr) {
                    seeds_out->push_back(i);
                    in_seed[static_cast<std::size_t>(i)] = 1;
                }
            }

            if (static_cast<int>(seeds_out->size()) < k_virtual) {
                for (int i : cand2) {
                    if (static_cast<int>(seeds_out->size()) >= k_virtual) {
                        break;
                    }
                    if (in_seed[static_cast<std::size_t>(i)]) {
                        continue;
                    }
                    seeds_out->push_back(i);
                    in_seed[static_cast<std::size_t>(i)] = 1;
                }
            }
        }

        void FillAnchorBarycenter(const std::vector<const float*>& Xp,
                                  int d,
                                  const int* nbrs_flat,
                                  const float* pdir_flat,
                                  const float* wsym_flat,
                                  int k_graph,
                                  int seed,
                                  int anchor_neighbor_k,
                                  float* __restrict dst,
                                  std::vector<std::pair<float, int>>* wj_scratch) {
            const float* xs = Xp[static_cast<std::size_t>(seed)];
#pragma omp simd
            for (int r = 0; r < d; ++r) {
                dst[r] = xs[r];
            }
            float sumw = 1.0f;

            const int* nbrs = nbrs_flat + static_cast<std::size_t>(seed) * k_graph;
            const float* pij_row = pdir_flat + static_cast<std::size_t>(seed) * k_graph;
            const float* w_row = (wsym_flat != nullptr)
                                     ? (wsym_flat + static_cast<std::size_t>(seed) * k_graph)
                                     : nullptr;

            wj_scratch->resize(static_cast<std::size_t>(k_graph));
            for (int t = 0; t < k_graph; ++t) {
                const int j = nbrs[t];
                const float w = (w_row != nullptr)
                                    ? w_row[t]
                                    : WSym(pij_row[t],
                                           GetP(nbrs_flat, pdir_flat, k_graph, j, seed));
                (*wj_scratch)[static_cast<std::size_t>(t)] = {w, j};
            }
            const int Kuse = std::min(std::max(0, anchor_neighbor_k), k_graph);
            if (Kuse > 0 && !wj_scratch->empty()) {
                std::partial_sort(wj_scratch->begin(), wj_scratch->begin() + Kuse, wj_scratch->end(),
                                  [](const auto& a, const auto& b) { return a.first > b.first; });
                for (int t = 0; t < Kuse; ++t) {
                    const float w = (*wj_scratch)[static_cast<std::size_t>(t)].first;
                    if (!(w > 0.0f)) {
                        continue;
                    }
                    const int j = (*wj_scratch)[static_cast<std::size_t>(t)].second;
                    const float* xj = Xp[static_cast<std::size_t>(j)];
                    sumw += w;
#pragma omp simd
                    for (int r = 0; r < d; ++r) {
                        dst[r] += w * xj[r];
                    }
                }
                const float invsum = 1.0f / sumw;
#pragma omp simd
                for (int r = 0; r < d; ++r) {
                    dst[r] *= invsum;
                }
            }
        }

        ColMajorMatrix<float> ReconstructAll(const std::vector<ColMajorMatrix<float>>& C,
                                             const ColMajorMatrix<FullCode>& B,
                                             const ColMajorMatrix<float>& a) {
            const int d = C.empty() ? 0 : C.front().rows;
            const int n = B.cols;
            const int m = static_cast<int>(C.size());
            ColMajorMatrix<float> out(d, n);
            std::fill(out.data.begin(), out.data.end(), 0.0f);

#pragma omp parallel for default(none) schedule(static) shared(C, B, a, out) firstprivate(d, n, m)
            for (int i = 0; i < n; ++i) {
                float* dst = out.Col(i);
                for (int l = 0; l < m; ++l) {
                    const int code = static_cast<int>(B(l, i));
                    const float coeff = a(l, i);
                    const float* center = C[static_cast<std::size_t>(l)].Col(code);
#pragma omp simd
                    for (int r = 0; r < d; ++r) {
                        dst[r] += coeff * center[r];
                    }
                }
            }
            return out;
        }

        [[maybe_unused]] ColMajorMatrix<float> ReconstructAllForcedRootConstRoot(
            const std::vector<ColMajorMatrix<float>>& C,
            int forced_root_code,
            const ColMajorMatrix<FullCode>& B,
            const ColMajorMatrix<float>& a) {
            const int d = C.empty() ? 0 : C.front().rows;
            const int n = B.cols;
            const int m = static_cast<int>(C.size());
            ColMajorMatrix<float> out(d, n);
            std::fill(out.data.begin(), out.data.end(), 0.0f);

            if (m <= 0 || n <= 0) {
                return out;
            }
            if (forced_root_code < 0 || C.empty() || forced_root_code >= C.front().cols) {
                // Caller error; return zeros rather than crashing.
                return out;
            }

#pragma omp parallel for default(none) schedule(static) shared(C, B, a, out) firstprivate(d, n, m, forced_root_code)
            for (int i = 0; i < n; ++i) {
                float* dst = out.Col(i);
                for (int l = 0; l < m; ++l) {
                    const int code = (l == 0) ? forced_root_code : static_cast<int>(B(l, i));
                    const float coeff = a(l, i);
                    const float* center = C[static_cast<std::size_t>(l)].Col(code);
#pragma omp simd
                    for (int r = 0; r < d; ++r) {
                        dst[r] += coeff * center[r];
                    }
                }
            }
            return out;
        }

        void GreedyInitAbsForcedRoot(const Precomp& precomp,
                                     const ColMajorMatrix<float>& xC,
                                     const std::vector<int>& forced_root,
                                     ColMajorMatrix<FullCode>* B,
                                     ColMajorMatrix<float>* a) {
            const int m = precomp.m;
            const int H = precomp.H;
            const int n = xC.cols;

            B->rows = m;
            B->cols = n;
            B->data.assign(static_cast<std::size_t>(m) * n, 0);
            a->rows = m;
            a->cols = n;
            a->data.assign(static_cast<std::size_t>(m) * n, 0.0f);

#pragma omp parallel default(none) shared(precomp, xC, forced_root, B, a) firstprivate(m, H, n)
            {
                std::vector<float> rC(static_cast<std::size_t>(H), 0.0f);
                std::vector<std::uint8_t> available(static_cast<std::size_t>(m), 1);

#pragma omp for schedule(static)
                for (int i = 0; i < n; ++i) {
                    const float* xCi = xC.Col(i);
                    std::memcpy(rC.data(), xCi, sizeof(float) * static_cast<std::size_t>(H));
                    std::fill(available.begin(), available.end(), 1);

                    const int root_code = forced_root[static_cast<std::size_t>(i)];
                    const int root_flat = precomp.offsets[0] + root_code;
                    const float invn_root = precomp.invnorm_flat[static_cast<std::size_t>(root_flat)];
                    const float alpha_root = rC[static_cast<std::size_t>(root_flat)] * invn_root * invn_root;
                    (*B)(0, i) = static_cast<FullCode>(root_code);
                    (*a)(0, i) = alpha_root;

                    const float* Gcol_root = precomp.G.Col(root_flat);
                    float* dst = rC.data();
#pragma omp simd
                    for (int p = 0; p < H; ++p) {
                        dst[p] -= alpha_root * Gcol_root[p];
                    }
                    available[0] = 0;

                    for (int t = 1; t < m; ++t) {
                        int best_flat = 0;
                        float best_abs = -std::numeric_limits<float>::infinity();
                        float best_adj = 0.0f;
                        for (int flat = 0; flat < H; ++flat) {
                            const int layer = precomp.flat_layer[static_cast<std::size_t>(flat)];
                            if (!available[static_cast<std::size_t>(layer)]) {
                                continue;
                            }
                            const float adj = dst[flat] * precomp.invnorm_flat[static_cast<std::size_t>(flat)];
                            const float aval = std::abs(adj);
                            if (aval > best_abs) {
                                best_abs = aval;
                                best_adj = adj;
                                best_flat = flat;
                            }
                        }
                        const int layer = precomp.flat_layer[static_cast<std::size_t>(best_flat)];
                        const int code = best_flat - precomp.offsets[layer];
                        const float invn = precomp.invnorm_flat[static_cast<std::size_t>(best_flat)];
                        const float alpha = best_adj * invn;

                        (*B)(layer, i) = static_cast<FullCode>(code);
                        (*a)(layer, i) = alpha;

                        const float* Gcol = precomp.G.Col(best_flat);
#pragma omp simd
                        for (int p = 0; p < H; ++p) {
                            dst[p] -= alpha * Gcol[p];
                        }
                        available[static_cast<std::size_t>(layer)] = 0;
                    }
                }
            }
        }

        bool EncodeCentersAbsForcedRoot(const CodebookPack& codebooks,
                                        const Precomp& precomp,
                                        const ColMajorMatrix<float>& centers,
                                        const std::vector<int>& forced_codes,
                                        int ils_iters,
                                        int icm_iters,
                                        int perturb_k,
                                        std::uint32_t seed,
                                        ColMajorMatrix<float>* X_virtual,
                                        ColMajorMatrix<FullCode>* B_virtual,
                                        ColMajorMatrix<float>* a_virtual,
                                        std::string* error) {
            if (!B_virtual || !a_virtual) {
                if (error) {
                    *error = "EncodeCentersAbsForcedRoot: null output.";
                }
                return false;
            }
            const int n = centers.cols;
            if (n <= 0) {
                if (X_virtual) {
                    *X_virtual = {};
                }
                *B_virtual = {};
                *a_virtual = {};
                return true;
            }
            if (static_cast<int>(forced_codes.size()) != n) {
                if (error) {
                    *error = "EncodeCentersAbsForcedRoot: forced code length mismatch.";
                }
                return false;
            }

            const bool prof_en = stlq::linkage::LinkageBuildProfileEnabled();
            auto* prof = prof_en ? stlq::linkage::GetLinkageBuildProfileTls() : nullptr;

            // Hot path: avoid repeated allocations while encoding UMAP centers (runs for all bad clusters).
            static thread_local ColMajorMatrix<float> xC;
            xC.rows = precomp.H;
            xC.cols = n;
            xC.data.resize(static_cast<std::size_t>(precomp.H) * static_cast<std::size_t>(n));
            const double t0_gemm = prof_en ? stlq::linkage::LinkageBuildWallNowS() : 0.0;
            Gemm(true, false, 1.0f, precomp.C_all, centers, 0.0f, &xC);
            if (prof) {
                prof->bad_umap_encode_gemm_s += stlq::linkage::LinkageBuildWallNowS() - t0_gemm;
            }

            const double t0_greedy = prof_en ? stlq::linkage::LinkageBuildWallNowS() : 0.0;
            GreedyInitAbsForcedRoot(precomp, xC, forced_codes, B_virtual, a_virtual);
            if (prof) {
                prof->bad_umap_encode_greedy_s += stlq::linkage::LinkageBuildWallNowS() - t0_greedy;
            }

            static thread_local std::vector<float> X_norm2;
            static thread_local std::vector<float> cost;
            X_norm2.clear();
            cost.clear();
            const double t0_icm = prof_en ? stlq::linkage::LinkageBuildWallNowS() : 0.0;
            DynamicIcmWithIlsAbsNoNormal(centers, precomp, xC,
                                         icm_iters, ils_iters, perturb_k,
                                         seed, B_virtual, a_virtual, &X_norm2, &cost,
                                         false);
            if (prof) {
                prof->bad_umap_encode_icm_s += stlq::linkage::LinkageBuildWallNowS() - t0_icm;
            }

            if (X_virtual) {
                const double t0_rec = prof_en ? stlq::linkage::LinkageBuildWallNowS() : 0.0;
                *X_virtual = ReconstructAll(codebooks.books, *B_virtual, *a_virtual);
                if (prof) {
                    prof->bad_umap_encode_recon_s += stlq::linkage::LinkageBuildWallNowS() - t0_rec;
                }
            }
            return true;
        }

#if defined(STLQ_ENABLE_CUDA)
        namespace
        {
            inline void ThrowIf(cudaError_t st, const char* what) {
                if (st == cudaSuccess) return;
                throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(st));
            }

            inline void ThrowIf(cublasStatus_t st, const char* what) {
                if (st == CUBLAS_STATUS_SUCCESS) return;
                throw std::runtime_error(std::string(what) + ": cublasStatus=" + std::to_string(static_cast<int>(st)));
            }

            template <typename T>
            struct DevBuf
            {
                T* ptr = nullptr;
                std::size_t cap = 0;
                ~DevBuf() { Reset(); }

                void Reset() {
                    if (ptr) {
                        cudaFree(ptr);
                        ptr = nullptr;
                    }
                    cap = 0;
                }

                void Ensure(std::size_t n) {
                    if (n <= cap) return;
                    Reset();
                    if (n == 0) return;
                    ThrowIf(cudaMalloc(&ptr, sizeof(T) * n), "cudaMalloc");
                    cap = n;
                }
            };

            struct UmapEncodeCudaTls
            {
                LinkageEncodeBatchCudaWorkspace* ws = nullptr;
                DevBuf<float> d_centers;
                DevBuf<float> d_xC;
                DevBuf<float> d_norm2;
                DevBuf<int> d_forced;
                DevBuf<std::uint64_t> d_sample_ids;
                DevBuf<FullCode> d_B;
                DevBuf<std::uint32_t> d_B_u32;
                DevBuf<float> d_a;
                DevBuf<float> d_cost;

                ~UmapEncodeCudaTls() {
                    if (ws) {
                        DestroyLinkageEncodeBatchCudaWorkspace(ws);
                        ws = nullptr;
                    }
                }
            };

            bool EncodeCentersAbsForcedRootCuda(const RuntimeConfig& runtime_cfg,
                                                CudaStreamKernelsPool* cuda_pool,
                                                const CodebookPack& codebooks,
                                                const Precomp& precomp,
                                                const ColMajorMatrix<float>& centers,
                                                const std::vector<int>& forced_codes,
                                                int ils_iters,
                                                int icm_iters,
                                                int perturb_k,
                                                std::uint32_t seed,
                                                ColMajorMatrix<float>* X_virtual,
                                                ColMajorMatrix<FullCode>* B_virtual,
                                                ColMajorMatrix<float>* a_virtual,
                                                std::string* error) {
                if (!cuda_pool || !runtime_cfg.use_cuda) {
                    if (error) *error = "EncodeCentersAbsForcedRootCuda: cuda disabled.";
                    return false;
                }
                const int d = centers.rows;
                const int n = centers.cols;
                const int m = precomp.m;
                if (n <= 0) {
                    if (X_virtual) *X_virtual = {};
                    *B_virtual = {};
                    *a_virtual = {};
                    return true;
                }
                if (static_cast<int>(forced_codes.size()) != n) {
                    if (error) *error = "EncodeCentersAbsForcedRootCuda: forced code length mismatch.";
                    return false;
                }

                CudaCtx* ctx = cuda_pool->TryAcquire();
                if (!ctx) {
                    // Pool exhausted: fall back to CPU without treating it as an error.
                    return EncodeCentersAbsForcedRoot(codebooks, precomp, centers, forced_codes,
                                                      ils_iters, icm_iters, perturb_k, seed,
                                                      X_virtual, B_virtual, a_virtual, error);
                }
                auto release_ctx = [&]() { cuda_pool->Release(ctx); };

                static thread_local UmapEncodeCudaTls tls;
                if (!tls.ws) {
                    tls.ws = CreateLinkageEncodeBatchCudaWorkspace();
                }

                // Compute xC on CPU (MKL/BLAS path) to match the CPU baseline exactly.
                // The UMAP-center encoding is sensitive to xC noise; using cuBLAS GEMM here can lead to stable drift.
                static thread_local ColMajorMatrix<float> xC;
                xC.rows = precomp.H;
                xC.cols = n;
                xC.data.resize(static_cast<std::size_t>(precomp.H) * static_cast<std::size_t>(n));
                const bool prof_en = stlq::linkage::LinkageBuildProfileEnabled();
                auto* prof = prof_en ? stlq::linkage::GetLinkageBuildProfileTls() : nullptr;
                const double t0_xc = prof_en ? stlq::linkage::LinkageBuildWallNowS() : 0.0;
                Gemm(true, false, 1.0f, precomp.C_all, centers, 0.0f, &xC);
                if (prof) {
                    prof->bad_umap_encode_gemm_s += stlq::linkage::LinkageBuildWallNowS() - t0_xc;
                }

                std::vector<float> norm2(static_cast<std::size_t>(n), 0.0f);
                for (int i = 0; i < n; ++i) {
                    const float* xi = centers.Col(i);
                    float s = 0.0f;
#pragma omp simd reduction(+:s)
                    for (int r = 0; r < d; ++r) {
                        s += xi[r] * xi[r];
                    }
                    norm2[static_cast<std::size_t>(i)] = s;
                }

                // CPU greedy init (exact baseline) and run GPU ICM/ILS starting from these codes.
                ColMajorMatrix<FullCode> B_init(m, n);
                ColMajorMatrix<float> a_init(m, n);
                const double t0_greedy = prof_en ? stlq::linkage::LinkageBuildWallNowS() : 0.0;
                GreedyInitAbsForcedRoot(precomp, xC, forced_codes, &B_init, &a_init);
                if (prof) {
                    prof->bad_umap_encode_greedy_s += stlq::linkage::LinkageBuildWallNowS() - t0_greedy;
                }

                try {
                    ThrowIf(cudaSetDevice(ctx->device), "cudaSetDevice");
                    tls.d_xC.Ensure(static_cast<std::size_t>(precomp.H) * static_cast<std::size_t>(n));
                    tls.d_norm2.Ensure(static_cast<std::size_t>(n));
                    tls.d_forced.Ensure(static_cast<std::size_t>(n));
                    tls.d_B.Ensure(static_cast<std::size_t>(m) * static_cast<std::size_t>(n));
                    tls.d_a.Ensure(static_cast<std::size_t>(m) * static_cast<std::size_t>(n));
                    tls.d_cost.Ensure(static_cast<std::size_t>(n));

                    ThrowIf(cudaMemcpyAsync(tls.d_xC.ptr,
                                            xC.data.data(),
                                            sizeof(float) * static_cast<std::size_t>(precomp.H) * static_cast<
                                                std::size_t>(n),
                                            cudaMemcpyHostToDevice,
                                            ctx->stream),
                            "cudaMemcpyAsync(xC)");
                    ThrowIf(cudaMemcpyAsync(tls.d_norm2.ptr,
                                            norm2.data(),
                                            sizeof(float) * static_cast<std::size_t>(n),
                                            cudaMemcpyHostToDevice,
                                            ctx->stream),
                            "cudaMemcpyAsync(norm2)");
                    ThrowIf(cudaMemcpyAsync(tls.d_forced.ptr,
                                            forced_codes.data(),
                                            sizeof(int) * static_cast<std::size_t>(n),
                                            cudaMemcpyHostToDevice,
                                            ctx->stream),
                            "cudaMemcpyAsync(forced_codes)");
                    ThrowIf(cudaMemcpyAsync(tls.d_B.ptr,
                                            B_init.data.data(),
                                            sizeof(FullCode) * static_cast<std::size_t>(m) * static_cast<std::size_t>(
                                                n),
                                            cudaMemcpyHostToDevice,
                                            ctx->stream),
                            "cudaMemcpyAsync(B_init)");
                    ThrowIf(cudaMemcpyAsync(tls.d_a.ptr,
                                            a_init.data.data(),
                                            sizeof(float) * static_cast<std::size_t>(m) * static_cast<std::size_t>(n),
                                            cudaMemcpyHostToDevice,
                                            ctx->stream),
                            "cudaMemcpyAsync(a_init)");

                    std::string local_err;
                    if (!LinkageEncodeBatchCudaForcedRootDeviceXcFromInitB(*ctx, precomp,
                                                                           tls.d_xC.ptr, tls.d_norm2.ptr,
                                                                           tls.d_forced.ptr,
                                                                           n, icm_iters, ils_iters, perturb_k,
                                                                           seed, /*sample_id_offset=*/0,
                                                                           tls.d_B.ptr, tls.d_a.ptr, tls.d_cost.ptr,
                                                                           tls.ws, &local_err)) {
                        if (error)
                            *error = local_err.empty()
                                         ? "EncodeCentersAbsForcedRootCuda: LinkageEncodeBatchCudaForcedRoot failed."
                                         : local_err;
                        release_ctx();
                        return false;
                    }

                    B_virtual->rows = m;
                    B_virtual->cols = n;
                    B_virtual->data.resize(static_cast<std::size_t>(m) * static_cast<std::size_t>(n));
                    a_virtual->rows = m;
                    a_virtual->cols = n;
                    a_virtual->data.resize(static_cast<std::size_t>(m) * static_cast<std::size_t>(n));

                    ThrowIf(cudaMemcpyAsync(B_virtual->data.data(),
                                            tls.d_B.ptr,
                                            sizeof(FullCode) * static_cast<std::size_t>(m) * static_cast<std::size_t>(
                                                n),
                                            cudaMemcpyDeviceToHost,
                                            ctx->stream),
                            "cudaMemcpyAsync(B)");
                    ThrowIf(cudaMemcpyAsync(a_virtual->data.data(),
                                            tls.d_a.ptr,
                                            sizeof(float) * static_cast<std::size_t>(m) * static_cast<std::size_t>(n),
                                            cudaMemcpyDeviceToHost,
                                            ctx->stream),
                            "cudaMemcpyAsync(a)");
                    ThrowIf(cudaStreamSynchronize(ctx->stream), "cudaStreamSynchronize(umap_encode)");

                    if (prof_en) {
                        if (prof) {
                            const CudaLinkageEncodeLastProfile lp = GetCudaLinkageEncodeLastProfile(tls.ws);
                            if (lp.valid) {
                                prof->bad_umap_encode_icm_s += (lp.ls_cost_s + lp.icm_s + lp.ils_s);
                            }
                        }
                        ConsumeCudaLinkageEncodeLastProfile(tls.ws);
                    }
                }
                catch (const std::exception& e) {
                    if (error) *error = e.what();
                    release_ctx();
                    return false;
                }

                release_ctx();

                if (X_virtual) {
                    const double t0_rec = prof_en ? stlq::linkage::LinkageBuildWallNowS() : 0.0;
                    *X_virtual = ReconstructAll(codebooks.books, *B_virtual, *a_virtual);
                    if (prof) {
                        prof->bad_umap_encode_recon_s += stlq::linkage::LinkageBuildWallNowS() - t0_rec;
                    }
                }

                // One-time CPU vs CUDA check (profiling-only) to help diagnose drift.
                // This is intentionally tiny: compare the first few centers of the first cluster we see.
                if (prof_en) {
                    static std::atomic<bool> s_once{false};
                    const bool do_once = !s_once.exchange(true, std::memory_order_relaxed);
                    if (do_once) {
                        // Prefer a full check when n is small (typical for SIFT1M bad clusters), otherwise sample a prefix.
                        const int n_cmp = (n <= 1024) ? n : std::min(n, 64);
                        ColMajorMatrix<float> centers_cmp(d, n_cmp);
                        for (int j = 0; j < n_cmp; ++j) {
                            std::memcpy(centers_cmp.Col(j), centers.Col(j),
                                        sizeof(float) * static_cast<std::size_t>(d));
                        }
                        std::vector<int> forced_cmp(forced_codes.begin(), forced_codes.begin() + n_cmp);

                        ColMajorMatrix<FullCode> B_cpu;
                        ColMajorMatrix<float> a_cpu;
                        std::string err_cpu;
                        if (EncodeCentersAbsForcedRoot(codebooks, precomp, centers_cmp, forced_cmp,
                                                       ils_iters, icm_iters, perturb_k, seed,
                                                       /*X_virtual=*/nullptr, &B_cpu, &a_cpu, &err_cpu)) {
                            ColMajorMatrix<float> xC_cmp(precomp.H, n_cmp);
                            for (int j = 0; j < n_cmp; ++j) {
                                std::memcpy(xC_cmp.Col(j),
                                            xC.Col(j),
                                            sizeof(float) * static_cast<std::size_t>(precomp.H));
                            }
                            std::vector<float> norm2_cmp(static_cast<std::size_t>(n_cmp));
                            for (int j = 0; j < n_cmp; ++j) {
                                norm2_cmp[static_cast<std::size_t>(j)] = norm2[static_cast<std::size_t>(j)];
                            }
                            std::vector<float> cost_cpu;
                            std::vector<float> cost_gpu;
                            ComputeCosts(precomp, xC_cmp, B_cpu, a_cpu, norm2_cmp, &cost_cpu);

                            ColMajorMatrix<FullCode> B_gpu(m, n_cmp);
                            ColMajorMatrix<float> a_gpu(m, n_cmp);
                            for (int j = 0; j < n_cmp; ++j) {
                                std::memcpy(B_gpu.Col(j), B_virtual->Col(j),
                                            sizeof(FullCode) * static_cast<std::size_t>(m));
                                std::memcpy(a_gpu.Col(j), a_virtual->Col(j),
                                            sizeof(float) * static_cast<std::size_t>(m));
                            }
                            ComputeCosts(precomp, xC_cmp, B_gpu, a_gpu, norm2_cmp, &cost_gpu);

                            std::uint64_t mism_codes = 0;
                            std::uint64_t mism_cols = 0;
                            for (int j = 0; j < n_cmp; ++j) {
                                bool col_mism = false;
                                for (int l = 0; l < m; ++l) {
                                    if (B_cpu(l, j) != B_gpu(l, j)) {
                                        mism_codes += 1;
                                        col_mism = true;
                                    }
                                }
                                if (col_mism) {
                                    mism_cols += 1;
                                }
                            }
                            double avg_cpu = 0.0, avg_gpu = 0.0;
                            double max_abs_diff = 0.0;
                            for (int j = 0; j < n_cmp; ++j) {
                                const auto c_cpu = static_cast<double>(cost_cpu[static_cast<std::size_t>(j)]);
                                const auto c_gpu = static_cast<double>(cost_gpu[static_cast<std::size_t>(j)]);
                                avg_cpu += c_cpu;
                                avg_gpu += c_gpu;
                                max_abs_diff = std::max(max_abs_diff, std::abs(c_cpu - c_gpu));
                            }
                            avg_cpu /= static_cast<double>(n_cmp);
                            avg_gpu /= static_cast<double>(n_cmp);
                            LogInfo("Bad UMAP encode check (" + std::to_string(n_cmp) +
                                " centers): B_mismatch_cols=" + std::to_string(mism_cols) +
                                " B_mismatch_codes=" + std::to_string(mism_codes) +
                                " avg_cost_cpu=" + std::to_string(avg_cpu) +
                                " avg_cost_cuda=" + std::to_string(avg_gpu) +
                                " max_abs_cost_diff=" + std::to_string(max_abs_diff));
                        }
                        else {
                            LogWarn("Bad UMAP encode check: CPU reference failed: " + err_cpu);
                        }
                    }
                }
                return true;
            }

            bool EncodeCentersAbsForcedRootCudaLargeRootConstRoot(const RuntimeConfig& runtime_cfg,
                                                                  CudaStreamKernelsPool* cuda_pool,
                                                                  const CodebookPack& codebooks,
                                                                  int forced_root_code,
                                                                  const ColMajorMatrix<float>& centers,
                                                                  int ils_iters,
                                                                  int icm_iters,
                                                                  int perturb_k,
                                                                  std::uint32_t seed,
                                                                  ColMajorMatrix<float>* X_virtual,
                                                                  ColMajorMatrix<FullCode>* B_virtual,
                                                                  ColMajorMatrix<float>* a_virtual,
                                                                  std::string* error) {
                if (!cuda_pool || !runtime_cfg.use_cuda) {
                    if (error) *error = "EncodeCentersAbsForcedRootCudaLargeRootConstRoot: cuda disabled.";
                    return false;
                }
                if (!B_virtual || !a_virtual) {
                    if (error) *error = "EncodeCentersAbsForcedRootCudaLargeRootConstRoot: null output.";
                    return false;
                }
                const int d = centers.rows;
                const int n = centers.cols;
                const int m = static_cast<int>(codebooks.books.size());
                if (n <= 0) {
                    if (X_virtual) *X_virtual = {};
                    *B_virtual = {};
                    *a_virtual = {};
                    return true;
                }
                if (d <= 0 || m <= 0 || static_cast<int>(codebooks.books.size()) != m) {
                    if (error) *error = "EncodeCentersAbsForcedRootCudaLargeRootConstRoot: invalid shape.";
                    return false;
                }
                if (codebooks.books.empty() || codebooks.books.front().cols <= 0) {
                    if (error) *error = "EncodeCentersAbsForcedRootCudaLargeRootConstRoot: invalid root codebook.";
                    return false;
                }
                const int h0 = codebooks.books.front().cols;
                if (forced_root_code < 0 || forced_root_code >= h0) {
                    if (error)
                        *error =
                            "EncodeCentersAbsForcedRootCudaLargeRootConstRoot: forced_root_code out of range.";
                    return false;
                }
                const bool need_root_u32 = (h0 > 256);

                CudaCtx* ctx = (runtime_cfg.cuda_linkage_wait_for_ctx ? cuda_pool->Acquire() : cuda_pool->TryAcquire());
                if (!ctx) {
                    if (error) {
                        *error = "EncodeCentersAbsForcedRootCudaLargeRootConstRoot: cuda ctx unavailable; "
                            "set runtime.cuda_linkage_wait_for_ctx=true and/or increase runtime.cuda_pool_size.";
                    }
                    return false;
                }
                auto release_ctx = [&]() { cuda_pool->Release(ctx); };

                static thread_local UmapEncodeCudaTls tls;
                if (!tls.ws) {
                    tls.ws = CreateLinkageEncodeBatchCudaWorkspace();
                }

                std::vector<float> norm2(static_cast<std::size_t>(n), 0.0f);
                for (int i = 0; i < n; ++i) {
                    const float* xi = centers.Col(i);
                    float s = 0.0f;
#pragma omp simd reduction(+:s)
                    for (int r = 0; r < d; ++r) {
                        s += xi[r] * xi[r];
                    }
                    norm2[static_cast<std::size_t>(i)] = s;
                }

                std::vector<std::uint64_t> sample_ids(static_cast<std::size_t>(n), 0);
                for (int i = 0; i < n; ++i) {
                    sample_ids[static_cast<std::size_t>(i)] =
                        (static_cast<std::uint64_t>(static_cast<std::uint32_t>(forced_root_code)) << 32) |
                        static_cast<std::uint32_t>(i);
                }

                try {
                    ThrowIf(cudaSetDevice(ctx->device), "cudaSetDevice");

                    tls.d_centers.Ensure(static_cast<std::size_t>(d) * static_cast<std::size_t>(n));
                    tls.d_norm2.Ensure(static_cast<std::size_t>(n));
                    tls.d_sample_ids.Ensure(static_cast<std::size_t>(n));
                    if (!need_root_u32) {
                        tls.d_B.Ensure(static_cast<std::size_t>(m) * static_cast<std::size_t>(n));
                    }
                    else {
                        tls.d_B_u32.Ensure(static_cast<std::size_t>(m) * static_cast<std::size_t>(n));
                    }
                    tls.d_a.Ensure(static_cast<std::size_t>(m) * static_cast<std::size_t>(n));
                    tls.d_cost.Ensure(static_cast<std::size_t>(n));

                    ThrowIf(cudaMemcpyAsync(tls.d_centers.ptr,
                                            centers.data.data(),
                                            sizeof(float) * static_cast<std::size_t>(d) * static_cast<std::size_t>(n),
                                            cudaMemcpyHostToDevice,
                                            ctx->stream),
                            "cudaMemcpyAsync(centers)");
                    ThrowIf(cudaMemcpyAsync(tls.d_norm2.ptr,
                                            norm2.data(),
                                            sizeof(float) * static_cast<std::size_t>(n),
                                            cudaMemcpyHostToDevice,
                                            ctx->stream),
                            "cudaMemcpyAsync(norm2)");
                    ThrowIf(cudaMemcpyAsync(tls.d_sample_ids.ptr,
                                            sample_ids.data(),
                                            sizeof(std::uint64_t) * static_cast<std::size_t>(n),
                                            cudaMemcpyHostToDevice,
                                            ctx->stream),
                            "cudaMemcpyAsync(sample_ids)");

                    std::string local_err;
                    if (!need_root_u32) {
                        if (!LinkageEncodeBatchCudaForcedRootWithSampleIdsLargeRootConstRoot(*ctx,
                            codebooks,
                            forced_root_code,
                            tls.d_centers.ptr,
                            tls.d_norm2.ptr,
                            tls.d_sample_ids.ptr,
                            n,
                            icm_iters,
                            ils_iters,
                            perturb_k,
                            seed,
                            tls.d_B.ptr,
                            tls.d_a.ptr,
                            tls.d_cost.ptr,
                            tls.ws,
                            &local_err)) {
                            if (error) *error = local_err;
                            release_ctx();
                            return false;
                        }
                    }
                    else {
                        if (!LinkageEncodeBatchCudaForcedRootWithSampleIdsLargeRootConstRootU32(*ctx,
                            codebooks,
                            forced_root_code,
                            tls.d_centers.ptr,
                            tls.d_norm2.ptr,
                            tls.d_sample_ids.ptr,
                            n,
                            icm_iters,
                            ils_iters,
                            perturb_k,
                            seed,
                            tls.d_B_u32.ptr,
                            tls.d_a.ptr,
                            tls.d_cost.ptr,
                            tls.ws,
                            &local_err)) {
                            if (error) *error = local_err;
                            release_ctx();
                            return false;
                        }
                    }

                    B_virtual->rows = m;
                    B_virtual->cols = n;
                    B_virtual->data.resize(static_cast<std::size_t>(m) * static_cast<std::size_t>(n));
                    a_virtual->rows = m;
                    a_virtual->cols = n;
                    a_virtual->data.resize(static_cast<std::size_t>(m) * static_cast<std::size_t>(n));

                    if (!need_root_u32) {
                        ThrowIf(cudaMemcpyAsync(B_virtual->data.data(),
                                                tls.d_B.ptr,
                                                sizeof(FullCode) * static_cast<std::size_t>(m) * static_cast<
                                                    std::size_t>(n),
                                                cudaMemcpyDeviceToHost,
                                                ctx->stream),
                                "cudaMemcpyAsync(B)");
                    }
                    else {
                        // Download u32 codes and cast layers 1..m-1 into the legacy FullCode matrix.
                        // Layer0 is implicitly `forced_root_code` and not representable in FullCode (u8) when h0>256.
                        static thread_local std::vector<std::uint32_t> tmp_B_u32;
                        tmp_B_u32.resize(static_cast<std::size_t>(m) * static_cast<std::size_t>(n));
                        ThrowIf(cudaMemcpyAsync(tmp_B_u32.data(),
                                                tls.d_B_u32.ptr,
                                                sizeof(std::uint32_t) * static_cast<std::size_t>(m) * static_cast<
                                                    std::size_t>(n),
                                                cudaMemcpyDeviceToHost,
                                                ctx->stream),
                                "cudaMemcpyAsync(B_u32)");
                        ThrowIf(cudaStreamSynchronize(ctx->stream),
                                "cudaStreamSynchronize(umap_encode_large_root_u32_codes)");
                        for (int i = 0; i < n; ++i) {
                            (*B_virtual)(0, i) = static_cast<FullCode>(0);
                            for (int l = 1; l < m; ++l) {
                                (*B_virtual)(l, i) =
                                    static_cast<FullCode>(tmp_B_u32[static_cast<std::size_t>(l) +
                                        static_cast<std::size_t>(m) * static_cast<std::size_t>(i)]);
                            }
                        }
                    }
                    ThrowIf(cudaMemcpyAsync(a_virtual->data.data(),
                                            tls.d_a.ptr,
                                            sizeof(float) * static_cast<std::size_t>(m) * static_cast<std::size_t>(n),
                                            cudaMemcpyDeviceToHost,
                                            ctx->stream),
                            "cudaMemcpyAsync(a)");
                    ThrowIf(cudaStreamSynchronize(ctx->stream), "cudaStreamSynchronize(umap_encode_large_root)");
                }
                catch (const std::exception& e) {
                    if (error) *error = e.what();
                    release_ctx();
                    return false;
                }

                release_ctx();

                if (X_virtual) {
                    *X_virtual = need_root_u32
                                     ? ReconstructAllForcedRootConstRoot(
                                         codebooks.books, forced_root_code, *B_virtual, *a_virtual)
                                     : ReconstructAll(codebooks.books, *B_virtual, *a_virtual);
                }
                return true;
            }
        } // namespace
#endif


        bool BuildUmapCentersForClusterFromKnnTables(const VirtualConfig& vcfg,
                                                     const ColMajorMatrix<float>& X,
                                                     const std::vector<int>& cluster_cols,
                                                     const std::vector<std::uint32_t>& knn_ids_flat,
                                                     const std::vector<float>& knn_dists_flat,
                                                     int k_graph_total,
                                                     int k_graph_umap,
                                                     int k_virtual,
                                                     int forced_root_code,
                                                     ColMajorMatrix<float>* centers,
                                                     std::vector<int>* forced_codes,
                                                     std::string* error) {
            if (!centers || !forced_codes) {
                if (error) {
                    *error = "BuildUmapCentersForClusterFromKnnTables: null output.";
                }
                return false;
            }
            const int d = X.rows;
            const int n_local = static_cast<int>(cluster_cols.size());
            if (d <= 0 || n_local <= 0 || k_virtual <= 0) {
                *centers = {};
                forced_codes->clear();
                return true;
            }
            k_virtual = std::min(k_virtual, n_local);
            k_graph_total = std::min(std::max(1, k_graph_total), n_local - 1);
            k_graph_umap = std::min(std::max(1, k_graph_umap), k_graph_total);
            if (static_cast<int>(knn_ids_flat.size()) != n_local * k_graph_total ||
                static_cast<int>(knn_dists_flat.size()) != n_local * k_graph_total) {
                if (error) {
                    *error = "BuildUmapCentersForClusterFromKnnTables: knn table shape mismatch.";
                }
                return false;
            }

            static thread_local UmapScratch scratch;

            scratch.Xp.resize(static_cast<std::size_t>(n_local));
            for (int i = 0; i < n_local; ++i) {
                const int gi = cluster_cols[static_cast<std::size_t>(i)];
                scratch.Xp[static_cast<std::size_t>(i)] = X.Col(gi);
            }

            const bool prof_en = stlq::linkage::LinkageBuildProfileEnabled();
            auto* prof = prof_en ? stlq::linkage::GetLinkageBuildProfileTls() : nullptr;

            const float target = std::log2(static_cast<float>(std::max(k_graph_umap, 2)));
            const double t0_rho_sigma = prof_en ? stlq::linkage::LinkageBuildWallNowS() : 0.0;
            UmapRhoSigmaStrided(knn_dists_flat.data(),
                                k_graph_total,
                                n_local,
                                k_graph_umap,
                                std::max(1, vcfg.local_connectivity),
                                target,
                                &scratch.rho,
                                &scratch.sigma);
            if (prof) {
                prof->bad_umap_rho_sigma_s += stlq::linkage::LinkageBuildWallNowS() - t0_rho_sigma;
            }

            const std::size_t flat_sz = static_cast<std::size_t>(n_local) * k_graph_umap;
            scratch.nbrs_flat.resize(flat_sz);
            scratch.pdir_flat.resize(flat_sz);
            const double t0_build_pdir = prof_en ? stlq::linkage::LinkageBuildWallNowS() : 0.0;
            for (int i = 0; i < n_local; ++i) {
                const std::uint32_t* idx_row =
                    knn_ids_flat.data() + static_cast<std::size_t>(i) * k_graph_total;
                const float* dist_row = knn_dists_flat.data() + static_cast<std::size_t>(i) * k_graph_total;
                const float rho_i = scratch.rho[static_cast<std::size_t>(i)];
                const float sigma_i = std::max(scratch.sigma[static_cast<std::size_t>(i)], kUmapEps);
                for (int t = 0; t < k_graph_umap; ++t) {
                    const int j = static_cast<int>(idx_row[t]);
                    const float dij = dist_row[t];
                    scratch.nbrs_flat[static_cast<std::size_t>(i) * k_graph_umap + t] = j;
                    scratch.pdir_flat[static_cast<std::size_t>(i) * k_graph_umap + t] = PIj(dij, rho_i, sigma_i);
                }
            }
            if (prof) {
                prof->bad_umap_build_pdir_s += stlq::linkage::LinkageBuildWallNowS() - t0_build_pdir;
            }

            scratch.deg.resize(static_cast<std::size_t>(n_local));
            scratch.wsym_flat.resize(flat_sz);
            const double t0_wsym_deg = prof_en ? stlq::linkage::LinkageBuildWallNowS() : 0.0;
            for (int i = 0; i < n_local; ++i) {
                float s = 0.0f;
                const int* nbrs = scratch.nbrs_flat.data() + static_cast<std::size_t>(i) * k_graph_umap;
                const float* pij_row = scratch.pdir_flat.data() + static_cast<std::size_t>(i) * k_graph_umap;
                float* w_row = scratch.wsym_flat.data() + static_cast<std::size_t>(i) * k_graph_umap;
                for (int t = 0; t < k_graph_umap; ++t) {
                    const int j = nbrs[t];
                    const float pij = pij_row[t];
                    const float pji = GetP(scratch.nbrs_flat.data(), scratch.pdir_flat.data(), k_graph_umap, j, i);
                    const float w = WSym(pij, pji);
                    w_row[t] = w;
                    s += w;
                }
                scratch.deg[static_cast<std::size_t>(i)] = s;
            }
            if (prof) {
                prof->bad_umap_wsym_deg_s += stlq::linkage::LinkageBuildWallNowS() - t0_wsym_deg;
            }

            const auto thr = static_cast<float>(vcfg.overlap_thr);
            const double t0_seed = prof_en ? stlq::linkage::LinkageBuildWallNowS() : 0.0;
            SelectSeedsUmapSimpleInto(scratch.nbrs_flat.data(),
                                      scratch.pdir_flat.data(),
                                      scratch.wsym_flat.data(),
                                      n_local,
                                      k_graph_umap,
                                      scratch.deg,
                                      k_virtual,
                                      thr,
                                      vcfg.prefer_peaks,
                                      &scratch,
                                      &scratch.seeds);
            if (prof) {
                prof->bad_umap_seed_select_s += stlq::linkage::LinkageBuildWallNowS() - t0_seed;
            }
            auto& seeds = scratch.seeds;
            if (static_cast<int>(seeds.size()) < k_virtual) {
                scratch.in_seed.resize(static_cast<std::size_t>(n_local));
                std::fill(scratch.in_seed.begin(), scratch.in_seed.end(), 0);
                for (int s : seeds) {
                    if (s >= 0 && s < n_local) {
                        scratch.in_seed[static_cast<std::size_t>(s)] = 1;
                    }
                }
                for (int i = 0; i < n_local && static_cast<int>(seeds.size()) < k_virtual; ++i) {
                    if (!scratch.in_seed[static_cast<std::size_t>(i)]) {
                        seeds.push_back(i);
                        scratch.in_seed[static_cast<std::size_t>(i)] = 1;
                    }
                }
                if (seeds.empty()) {
                    // Should be unreachable when n_local>0 and k_virtual>0; keep deterministic fallback.
                    for (int i = 0; i < std::min(k_virtual, n_local); ++i) {
                        seeds.push_back(i);
                    }
                }
            }

            ColMajorMatrix<float> centers_out(d, static_cast<int>(seeds.size()));
            std::vector<int> forced_out(static_cast<std::size_t>(seeds.size()), forced_root_code);

            const double t0_bary = prof_en ? stlq::linkage::LinkageBuildWallNowS() : 0.0;
            for (int s_idx = 0; s_idx < static_cast<int>(seeds.size()); ++s_idx) {
                const int seed = seeds[static_cast<std::size_t>(s_idx)];
                FillAnchorBarycenter(scratch.Xp, d,
                                     scratch.nbrs_flat.data(),
                                     scratch.pdir_flat.data(),
                                     scratch.wsym_flat.data(),
                                     k_graph_umap,
                                     seed,
                                     std::max(0, vcfg.anchor_neighbor_k),
                                     centers_out.Col(s_idx),
                                     &scratch.wj);
            }
            if (prof) {
                prof->bad_umap_barycenter_s += stlq::linkage::LinkageBuildWallNowS() - t0_bary;
            }

            *centers = std::move(centers_out);
            *forced_codes = std::move(forced_out);
            return true;
        }
    } // namespace

    int ResolveVirtualCount(int n_local, const VirtualConfig& cfg) {
        if (n_local <= 0) {
            return 0;
        }
        int k = cfg.use_fixed_virtual_per_cluster
                    ? cfg.fixed_virtual_per_cluster
                    : static_cast<int>(std::llround(cfg.virtual_ratio * static_cast<double>(n_local)));
        k = std::max(k, cfg.min_virtual);
        k = std::min(k, cfg.max_virtual);
        k = std::min(k, n_local);
        return std::max(0, k);
    }

    bool AddVirtualRootsUmapReencodeClusterFromKnnTables(const VirtualConfig& vcfg,
                                                         const CodebookPack& codebooks,
                                                         const Precomp& precomp,
                                                         const RuntimeConfig* runtime_cfg,
                                                         CudaStreamKernelsPool* cuda_pool,
                                                         const ColMajorMatrix<float>& X,
                                                         const std::vector<int>& cluster_cols,
                                                         const std::vector<std::uint32_t>& knn_ids_flat,
                                                         const std::vector<float>& knn_dists_flat,
                                                         int k_graph_total,

                                                         int k_graph_umap,
                                                         int forced_root_code,
                                                         int k_virtual,
                                                         int ils_iters,
                                                         int icm_iters,
                                                         int perturb_k,
                                                         std::uint32_t seed,
                                                         ColMajorMatrix<float>* X_virtual,
                                                         ColMajorMatrix<FullCode>* B_virtual,
                                                         ColMajorMatrix<float>* a_virtual,
                                                         std::string* error) {
        (void)runtime_cfg;
        (void)cuda_pool;
        if (!X_virtual || !B_virtual || !a_virtual) {
            if (error) {
                *error = "AddVirtualRootsUmapReencodeClusterFromKnnTables: null output.";
            }
            return false;
        }
        if (k_virtual <= 0) {
            *X_virtual = {};
            *B_virtual = {};
            *a_virtual = {};
            return true;
        }
        ColMajorMatrix<float> centers;
        std::vector<int> forced_codes;
        if (!BuildUmapCentersForClusterFromKnnTables(vcfg, X, cluster_cols,
                                                     knn_ids_flat, knn_dists_flat, k_graph_total, k_graph_umap,
                                                     k_virtual, forced_root_code,
                                                     &centers, &forced_codes, error)) {
            return false;
        }
        if (centers.cols <= 0) {
            *X_virtual = {};
            *B_virtual = {};
            *a_virtual = {};
            return true;
        }
        const bool prof_en = stlq::linkage::LinkageBuildProfileEnabled();
        auto* prof = prof_en ? stlq::linkage::GetLinkageBuildProfileTls() : nullptr;
        const double t0_encode = prof_en ? stlq::linkage::LinkageBuildWallNowS() : 0.0;
        bool ok = false;
#if defined(STLQ_ENABLE_CUDA)
        if (runtime_cfg && runtime_cfg->use_cuda && cuda_pool) {
            const bool have_full_precomp =
                (!precomp.G.data.empty() && !precomp.C_all.data.empty() && precomp.H > 0 && precomp.m > 0);
            if (have_full_precomp) {
                ok = EncodeCentersAbsForcedRootCuda(*runtime_cfg, cuda_pool,
                                                    codebooks, precomp, centers, forced_codes,
                                                    ils_iters, icm_iters, perturb_k, seed,
                                                    X_virtual, B_virtual, a_virtual, error);
            }
            else {
                ok = EncodeCentersAbsForcedRootCudaLargeRootConstRoot(*runtime_cfg, cuda_pool,
                                                                      codebooks, forced_root_code, centers,
                                                                      ils_iters, icm_iters, perturb_k, seed,
                                                                      X_virtual, B_virtual, a_virtual, error);
            }
        }
        else
#endif
        {
            if (precomp.G.data.empty()) {
                // CPU-only large pipeline can legitimately run with meta-only precomp (no G).
                // In that case, we cannot encode UMAP centers on CPU (the baseline encoder requires full-precomp).
                // Treat this as a non-fatal feature disable: skip virtual roots for this cluster.
                static std::atomic<bool> did_warn{false};
                bool expected = false;
                if (did_warn.compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
                    LogWarn(
                        "UMAP virtual roots: skipped because full-precomp(G) is unavailable and CUDA is disabled. "
                        "To enable UMAP virtual roots, build/enable CUDA (runtime.use_cuda=1) or disable virtual nodes (virtual.enabled=0).");
                }
                *X_virtual = {};
                *B_virtual = {};
                *a_virtual = {};
                if (error) {
                    *error =
                        "AddVirtualRootsUmapReencodeClusterFromKnnTables: skipped (missing full-precomp G; CUDA disabled).";
                }
                return true;
            }
            ok = EncodeCentersAbsForcedRoot(codebooks, precomp, centers, forced_codes,
                                            ils_iters, icm_iters, perturb_k, seed,
                                            X_virtual, B_virtual, a_virtual, error);
        }
        if (prof) {
            const double dt = stlq::linkage::LinkageBuildWallNowS() - t0_encode;
            prof->bad_umap_encode_s += dt;
            if (dt > prof->bad_umap_encode_max_cluster_s) {
                // Here forced_root_code is the cluster id (cid) in all current callers.
                prof->bad_umap_encode_max_cluster_s = dt;
                prof->bad_umap_encode_max_cluster_cid = forced_root_code;
                prof->bad_umap_encode_max_cluster_n_real = static_cast<int>(cluster_cols.size());
                prof->bad_umap_encode_max_cluster_k_virtual = k_virtual;
                prof->bad_umap_encode_max_cluster_n_centers = centers.cols;
            }
        }
        return ok;
    }

    bool AddVirtualNodes(const Config& config,
                         const CodebookPack& codebooks,
                         const Precomp& precomp,
                         const ColMajorMatrix<float>& X_real,
                         const BaseEncoding& base_real,
                         const std::vector<bool>& is_bad_cluster,
                         VirtualEncoding* virt_out,
                         BadClusterKnnCache* knn_cache_out,
                         std::string* error) {
        if (virt_out == nullptr) {
            if (error) {
                *error = "AddVirtualNodes received null pointers.";
            }
            return false;
        }
        if (!config.virtual_cfg.enabled) {
            *virt_out = {};
            if (knn_cache_out) {
                knn_cache_out->Clear();
            }
            return true;
        }

        const int d = X_real.rows;
        const int n_real = base_real.B.cols;
        const int m = base_real.B.rows;
        if (d <= 0 || n_real <= 0 || m <= 0) {
            *virt_out = {};
            if (knn_cache_out) {
                knn_cache_out->Clear();
            }
            return true;
        }
        if (static_cast<int>(codebooks.books.size()) != m) {
            if (error) {
                *error = "AddVirtualNodes: codebook count mismatch.";
            }
            return false;
        }
        const int nlist = codebooks.books.empty() ? 0 : codebooks.books.front().cols;
        if (nlist <= 0) {
            if (error) {
                *error = "AddVirtualNodes: invalid root codebook size.";
            }
            return false;
        }
        if (static_cast<int>(is_bad_cluster.size()) != nlist) {
            if (error) {
                *error = "AddVirtualNodes: is_bad_cluster is missing or size mismatch.";
            }
            return false;
        }

        std::vector<std::vector<int>> cluster_cols(static_cast<std::size_t>(nlist));
        for (int i = 0; i < n_real; ++i) {
            const int cid = static_cast<int>(base_real.B(0, i));
            cluster_cols[static_cast<std::size_t>(cid)].push_back(i);
        }

        int est_total_virtual = 0;
        std::vector<int> k_per_cluster(static_cast<std::size_t>(nlist), 0);
        std::vector<int> bad_cids;
        bad_cids.reserve(static_cast<std::size_t>(nlist));
        for (int cid = 0; cid < nlist; ++cid) {
            if (!is_bad_cluster[static_cast<std::size_t>(cid)]) {
                continue;
            }
            const int n_local = static_cast<int>(cluster_cols[static_cast<std::size_t>(cid)].size());
            if (n_local < 2) {
                continue;
            }
            const int k_virtual = ResolveVirtualCount(n_local, config.virtual_cfg);
            if (k_virtual <= 0) {
                continue;
            }
            k_per_cluster[static_cast<std::size_t>(cid)] = k_virtual;
            est_total_virtual += k_virtual;
            bad_cids.push_back(cid);
        }
        // Scheduling-only optimization: process the largest bad clusters first to reduce tail latency
        // (wall=max-thread time) under `#pragma omp for schedule(dynamic)`.
        // This must not affect semantics: all results are keyed by `cid`, and UMAP seed selection here
        // is deterministic (no global RNG coupling between clusters).
        std::sort(bad_cids.begin(), bad_cids.end(), [&](int a, int b)
        {
            const int ka = k_per_cluster[static_cast<std::size_t>(a)];
            const int kb = k_per_cluster[static_cast<std::size_t>(b)];
            if (ka != kb) return ka > kb;
            const int na = static_cast<int>(cluster_cols[static_cast<std::size_t>(a)].size());
            const int nb = static_cast<int>(cluster_cols[static_cast<std::size_t>(b)].size());
            if (na != nb) return na > nb;
            return a < b;
        });
        if (est_total_virtual <= 0) {
            LogInfo("UMAP virtual roots: none selected.");
            *virt_out = {};
            if (knn_cache_out) {
                knn_cache_out->Clear();
            }
            return true;
        }
        LogInfo("UMAP virtual roots: estimated centers to encode = " + std::to_string(est_total_virtual));

        VirtualEncoding virt;
        virt.m = m;
        virt.nlist = nlist;
        virt.B_by_cluster.assign(static_cast<std::size_t>(nlist), {});
        virt.a_by_cluster.assign(static_cast<std::size_t>(nlist), {});
        int virt_cols = 0;
        for (int cid = 0; cid < nlist; ++cid) {
            virt_cols += k_per_cluster[static_cast<std::size_t>(cid)];
        }

        BadClusterKnnCache cache;
        if (knn_cache_out) {
            cache.nlist = nlist;
            cache.knn_k_by_cluster.assign(static_cast<std::size_t>(nlist), 0);
            cache.knn_ids_by_cluster.resize(static_cast<std::size_t>(nlist));
        }

        std::atomic<int> done{0};
        std::atomic<bool> ok{true};
        std::string first_error;
        std::atomic_flag has_error = ATOMIC_FLAG_INIT;

#pragma omp parallel default(none) shared(codebooks, precomp, X_real, base_real, config, cluster_cols, bad_cids, k_per_cluster, virt, cache, knn_cache_out, done, ok, first_error, has_error) firstprivate(d, n_real, m, nlist)
        {
            ColMajorMatrix<float> centers_c;
            std::vector<int> forced_c;
            std::vector<std::uint32_t> idxs_flat;
            std::vector<float> dists_flat;
            std::string local_error;

#pragma omp for schedule(dynamic)
            for (int bi = 0; bi < static_cast<int>(bad_cids.size()); ++bi) {
                if (!ok.load(std::memory_order_relaxed)) {
                    continue;
                }

                const int cid = bad_cids[static_cast<std::size_t>(bi)];
                const int k_virtual = k_per_cluster[static_cast<std::size_t>(cid)];
                const auto& cols = cluster_cols[static_cast<std::size_t>(cid)];
                const int n_local = static_cast<int>(cols.size());
                if (n_local < 2 || k_virtual <= 0) {
                    int dd = done.fetch_add(1, std::memory_order_relaxed) + 1;
                    if (dd % 50 == 0 || dd == static_cast<int>(bad_cids.size())) {
                        PrintAugmentProgress(dd, static_cast<int>(bad_cids.size()));
                    }
                    continue;
                }

                const int mult_bad = std::max(1, config.hnsw.candidate_multiplier_bad);
                int knn_real = std::max(std::max(1, config.virtual_cfg.umap_knn_k),
                                        mult_bad * std::max(1, config.base.linkage.knn_k));
                knn_real = std::min(knn_real, n_local - 1);
                const int knn_umap = std::min(std::max(1, config.virtual_cfg.umap_knn_k), knn_real);

                forced_c.clear();
                dists_flat.clear();
                local_error.clear();

                std::vector<std::uint32_t>* ids_out = &idxs_flat;
                if (knn_cache_out) {
                    ids_out = &cache.knn_ids_by_cluster[static_cast<std::size_t>(cid)];
                    ids_out->clear();
                }
                else {
                    idxs_flat.clear();
                }

                if (!BuildClusterKnnHnswImpl(X_real, cols, config.hnsw,
                                             knn_real,
                                             ids_out, &dists_flat, &local_error)) {
                    ok.store(false, std::memory_order_relaxed);
                    if (!has_error.test_and_set(std::memory_order_relaxed)) {
                        first_error = local_error.empty() ? "HNSW KNN build failed." : local_error;
                    }
                    continue;
                }

                if (!BuildUmapCentersForClusterFromKnnTables(config.virtual_cfg,
                                                             X_real,
                                                             cols,
                                                             *ids_out,
                                                             dists_flat,
                                                             knn_real,
                                                             knn_umap,
                                                             k_virtual,
                                                             cid,
                                                             &centers_c,
                                                             &forced_c,
                                                             &local_error)) {
                    ok.store(false, std::memory_order_relaxed);
                    if (!has_error.test_and_set(std::memory_order_relaxed)) {
                        first_error = local_error.empty() ? "UMAP centers build failed." : local_error;
                    }
                    continue;
                }

                // Distances are no longer needed after UMAP center selection.
                dists_flat.clear();

                ColMajorMatrix<FullCode> B_virtual;
                ColMajorMatrix<float> a_virtual;
                const bool prof_en2 = stlq::linkage::LinkageBuildProfileEnabled();
                auto* prof2 = prof_en2 ? stlq::linkage::GetLinkageBuildProfileTls() : nullptr;
                const double t0_encode_cluster = prof_en2 ? stlq::linkage::LinkageBuildWallNowS() : 0.0;
                if (!EncodeCentersAbsForcedRoot(codebooks, precomp, centers_c, forced_c,
                                                config.base.encode.ils_iters,
                                                config.base.encode.icm_iters,
                                                config.base.encode.perturb_k,
                                                static_cast<std::uint32_t>(config.base.encode.seed),
                                                nullptr, &B_virtual, &a_virtual, &local_error)) {
                    ok.store(false, std::memory_order_relaxed);
                    if (!has_error.test_and_set(std::memory_order_relaxed)) {
                        first_error = local_error.empty() ? "Virtual encoding failed." : local_error;
                    }
                    continue;
                }
                if (prof2) {
                    const double dt = stlq::linkage::LinkageBuildWallNowS() - t0_encode_cluster;
                    if (dt > prof2->bad_umap_encode_max_cluster_s) {
                        prof2->bad_umap_encode_max_cluster_s = dt;
                        prof2->bad_umap_encode_max_cluster_cid = cid;
                        prof2->bad_umap_encode_max_cluster_n_real = n_local;
                        prof2->bad_umap_encode_max_cluster_k_virtual = k_virtual;
                        prof2->bad_umap_encode_max_cluster_n_centers = centers_c.cols;
                    }
                }

                virt.B_by_cluster[static_cast<std::size_t>(cid)] = std::move(B_virtual);
                virt.a_by_cluster[static_cast<std::size_t>(cid)] = std::move(a_virtual);
                if (knn_cache_out) {
                    cache.knn_k_by_cluster[static_cast<std::size_t>(cid)] = knn_real;
                }

                int dd = done.fetch_add(1, std::memory_order_relaxed) + 1;
                if (dd % 50 == 0 || dd == static_cast<int>(bad_cids.size())) {
                    PrintAugmentProgress(dd, static_cast<int>(bad_cids.size()));
                }
            }
        }

        if (!ok.load(std::memory_order_relaxed)) {
            if (error) {
                *error = first_error.empty() ? "AddVirtualNodes failed." : first_error;
            }
            return false;
        }

        LogInfo("UMAP virtual roots: encoded virtual=" + std::to_string(virt_cols) +
            " (real=" + std::to_string(n_real) + ")");
        *virt_out = std::move(virt);
        if (knn_cache_out) {
            *knn_cache_out = std::move(cache);
        }
        return true;
    }
} // namespace stlq
