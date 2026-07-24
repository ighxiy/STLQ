#include "stlq/quantizer/spkmeans.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "stlq/pipeline/app_utils.h"
#include "stlq/core/kernel_provider.h"
#include "stlq/core/kernel_provider_cuda_stream.h"
#include "stlq/io/col_block_reader.h"
#include "stlq/common/logger.h"
#include "stlq/common/timer.h"

namespace stlq
{
    namespace
    {
        constexpr float kEps = 1e-6f;

        inline void PopulateKmeansTimingFlags(KmeansTiming* timing,
                                              const KmeansConfig& cfg,
                                              int K) {
            if (!timing) return;
            const bool anneal_enabled = (cfg.annealing_factor > 0.0f) && (cfg.warmup_iters < cfg.max_iters);
            // Note: K-means timing can span multiple per-layer k-means calls (RVQ hierarchical init).
            // These fields should be interpreted as "enabled/used in ANY layer", not "last layer wins".
            timing->anneal_enabled = timing->anneal_enabled || (anneal_enabled ? 1 : 0);
            timing->anneal_no_spill = timing->anneal_no_spill || ((anneal_enabled && cfg.anneal_no_spill) ? 1 : 0);

            timing->hier2_enable = timing->hier2_enable || (cfg.large_k_hier2_enable ? 1 : 0);
            timing->hier2_train = timing->hier2_train || (cfg.large_k_hier2_train ? 1 : 0);
            timing->hier2_encode = timing->hier2_encode || (cfg.large_k_hier2_encode ? 1 : 0);
            timing->hier2_threshold = std::max(timing->hier2_threshold, cfg.large_k_threshold);
            timing->hier2_topL = std::max(timing->hier2_topL, cfg.large_k_top_coarse);

            timing->hier2_K = std::max(timing->hier2_K, K);
            // `used_hier2`, `hier2_K1`, `hier2_K2` are set only when a hier2 assignment path is taken.
            // Do NOT reset them here, otherwise later small-K layers would erase the "used" evidence.
        }

        struct Hier2Split
        {
            int K = 0; // requested
            int K1 = 0; // coarse
            int K2 = 0; // per-coarse fine capacity (ceil)
        };

        inline int RoundDownPow2(int x) {
            int p = 1;
            while ((p << 1) > 0 && (p << 1) <= x) p <<= 1;
            return p;
        }

        inline int RoundUpPow2(int x) {
            int p = 1;
            while (p > 0 && p < x) p <<= 1;
            return p;
        }

        inline Hier2Split ComputeHier2Split(int K, const KmeansConfig& cfg) {
            Hier2Split s;
            s.K = K;
            if (K <= 0) return s;

            int K1 = cfg.large_k_fixed_k1;
            if (K1 <= 0) {
                const double root = std::sqrt(static_cast<double>(K));
                K1 = static_cast<int>(std::lround(root));
                K1 = std::max(cfg.large_k_k1_min, std::min(cfg.large_k_k1_max, K1));
                if (cfg.large_k_k1_pow2) {
                    const int down = RoundDownPow2(K1);
                    const int up = RoundUpPow2(K1);
                    K1 = (std::abs(up - K1) <= std::abs(K1 - down)) ? up : down;
                    K1 = std::max(cfg.large_k_k1_min, std::min(cfg.large_k_k1_max, K1));
                }
                if (K1 > K) K1 = K;
            }
            else {
                K1 = std::max(1, std::min(K, K1));
            }

            const int K2 = (K + K1 - 1) / K1; // ceil(K/K1)
            s.K1 = K1;
            s.K2 = std::max(1, K2);
            return s;
        }

        inline int FlatIdFromHier2(int coarse, int fine_local, int K2) {
            return coarse * K2 + fine_local;
        }

        inline bool ShouldUseHier2(int K, const KmeansConfig& cfg) {
            return cfg.large_k_hier2_enable && (K >= cfg.large_k_threshold);
        }

        inline int CoarseGroupSize(int coarse, int K, int K2) {
            const int start = coarse * K2;
            const int remain = K - start;
            if (remain <= 0) return 0;
            return std::min(K2, remain);
        }

        ColMajorMatrix<float> BuildCoarseFromFineCPU(const ColMajorMatrix<float>& fine,
                                                     int K,
                                                     int K1,
                                                     int K2) {
            const int d = fine.rows;
            ColMajorMatrix<float> coarse(d, K1);
            coarse.data.assign(static_cast<std::size_t>(d) * static_cast<std::size_t>(K1), 0.0f);

#pragma omp parallel default(none) shared(coarse, fine) firstprivate(d, K, K1, K2, kEps)
            {
                std::vector<float> sum(static_cast<std::size_t>(d), 0.0f);
#pragma omp for schedule(static)
                for (int c = 0; c < K1; ++c) {
                    float* out = coarse.Col(c);
                    const int sz = CoarseGroupSize(c, K, K2);
                    if (sz <= 0) continue;

                    std::fill(sum.begin(), sum.end(), 0.0f);
                    for (int j = 0; j < sz; ++j) {
                        const int fid = FlatIdFromHier2(c, j, K2);
                        const float* f = fine.Col(fid);
#pragma omp simd
                        for (int r = 0; r < d; ++r) sum[static_cast<std::size_t>(r)] += f[r];
                    }

                    float ss = 0.0f;
#pragma omp simd reduction(+:ss)
                    for (int r = 0; r < d; ++r)
                        ss += sum[static_cast<std::size_t>(r)] * sum[static_cast<std::size_t>(
                            r)];
                    float n = std::sqrt(std::max(ss, kEps));
                    const float inv = (n > kEps) ? (1.0f / n) : 1.0f;
#pragma omp simd
                    for (int r = 0; r < d; ++r) out[r] = sum[static_cast<std::size_t>(r)] * inv;
                }
            }
            return coarse;
        }

        void NormalizeColumnsInPlace(ColMajorMatrix<float>* X) {
            if (!X) return;
            const int d = X->rows;
            const int n = X->cols;
            if (d <= 0 || n <= 0) return;
#pragma omp parallel for default(none) shared(X) firstprivate(d, n, kEps) schedule(static)
            for (int i = 0; i < n; ++i) {
                float* xi = X->Col(i);
                float norm = 0.0f;
#pragma omp simd reduction(+:norm)
                for (int r = 0; r < d; ++r) {
                    norm += xi[r] * xi[r];
                }
                norm = std::sqrt(std::max(norm, kEps));
                if (norm > kEps) {
                    const float inv = 1.0f / norm;
#pragma omp simd
                    for (int r = 0; r < d; ++r) {
                        xi[r] *= inv;
                    }
                }
            }
        }

        ColMajorMatrix<float> NormalizeColumns(const ColMajorMatrix<float>& X) {
            // Julia reference: normalize_data_fast (fast_spkmeans.jl:188-213).
            const int d = X.rows;
            const int n = X.cols;
            ColMajorMatrix<float> out(d, n);
#pragma omp parallel for default(none) shared(X, out) firstprivate(d, n, kEps) schedule(static)
            for (int i = 0; i < n; ++i) {
                const float* xi = X.Col(i);
                float* dst = out.Col(i);
                float norm = 0.0f;
#pragma omp simd reduction(+:norm)
                for (int r = 0; r < d; ++r) {
                    norm += xi[r] * xi[r];
                }
                norm = std::sqrt(std::max(norm, kEps));
                if (norm > kEps) {
#pragma omp simd
                    for (int r = 0; r < d; ++r) {
                        dst[r] = xi[r] / norm;
                    }
                }
                else {
                    std::copy(xi, xi + d, dst);
                }
            }
            return out;
        }

        ColMajorMatrix<float> InitializeCentersKmeansPP(const ColMajorMatrix<float>& X,
                                                        int k,
                                                        std::mt19937* rng) {
            // Julia reference: initialize_centers_kmeanspp_subset_accurate (fast_spkmeans.jl:55-106).
            const int d = X.rows;
            const int n = X.cols;
            ColMajorMatrix<float> centers(d, k);
            if (n == 0 || k == 0) {
                return centers;
            }

            std::uniform_int_distribution<int> uni(0, n - 1);
            int first_idx = uni(*rng);
            std::copy(X.Col(first_idx), X.Col(first_idx) + d, centers.Col(0));

            // Performance: incremental kmeans++ update avoids O(k^2) dot computations.
            std::vector<float> best_dot(n, -std::numeric_limits<float>::infinity());
            std::vector<float> distances(n, 0.0f);

            // Init best distance to first center.
            const float* c0 = centers.Col(0);
#pragma omp parallel for default(none) shared(X, c0, best_dot, distances) firstprivate(d, n, kEps) schedule(static)
            for (int i = 0; i < n; ++i) {
                const float* xi = X.Col(i);
                float dot = 0.0f;
#pragma omp simd reduction(+:dot)
                for (int r = 0; r < d; ++r) {
                    dot += xi[r] * c0[r];
                }
                best_dot[i] = dot;
                distances[i] = std::max(0.0f, 2.0f - 2.0f * dot);
            }

            for (int c = 1; c < k; ++c) {
                // Determinism: do not parallel-reduce `total`. Tiny rounding differences in `total`
                // can change the random threshold and pick a different center, causing large MSE drift
                // when OpenMP thread count changes.
                double total = 0.0;
                for (int i = 0; i < n; ++i) {
                    total += static_cast<double>(distances[i]);
                }

                int selected = uni(*rng);
                if (total > static_cast<double>(kEps)) {
                    std::uniform_real_distribution<double> ru(0.0, total);
                    double r = ru(*rng);
                    double csum = 0.0;
                    for (int i = 0; i < n; ++i) {
                        csum += static_cast<double>(distances[i]);
                        if (csum >= r) {
                            selected = i;
                            break;
                        }
                    }
                }
                std::copy(X.Col(selected), X.Col(selected) + d, centers.Col(c));

                const float* cnew = centers.Col(c);
#pragma omp parallel for default(none) shared(X, cnew, best_dot, distances) firstprivate(d, n, kEps) schedule(static)
                for (int i = 0; i < n; ++i) {
                    const float* xi = X.Col(i);
                    float dot = 0.0f;
#pragma omp simd reduction(+:dot)
                    for (int r = 0; r < d; ++r) {
                        dot += xi[r] * cnew[r];
                    }
                    if (dot > best_dot[i]) {
                        best_dot[i] = dot;
                        distances[i] = std::max(0.0f, 2.0f - 2.0f * dot);
                    }
                }
            }
            return centers;
        }

        ColMajorMatrix<float> InitializeCentersFast(const ColMajorMatrix<float>& X,
                                                    int k,
                                                    const KmeansConfig& cfg,
                                                    std::mt19937* rng) {
            // Julia reference: initialize_centers_fast (fast_spkmeans.jl:11-48).
            const int d = X.rows;
            const int n = X.cols;
            ColMajorMatrix<float> centers(d, k);
            if (n == 0 || k == 0) {
                return centers;
            }
            if (cfg.init_method == "kmeans++") {
                int sample_n = std::min(n, std::max(1, cfg.init_samples));
                if (sample_n == n) {
                    // Avoid an unnecessary shuffle/copy when using the full dataset for kmeans++ init.
                    centers = InitializeCentersKmeansPP(X, k, rng);
                }
                else {
                    ColMajorMatrix<float> sample(d, sample_n);
                    std::vector<int> indices(n);
                    std::iota(indices.begin(), indices.end(), 0);
                    std::shuffle(indices.begin(), indices.end(), *rng);
#pragma omp parallel for default(none) shared(X, indices, sample) firstprivate(d, sample_n, kEps) schedule(static)
                    for (int i = 0; i < sample_n; ++i) {
                        std::copy(X.Col(indices[i]), X.Col(indices[i]) + d, sample.Col(i));
                    }
                    centers = InitializeCentersKmeansPP(sample, k, rng);
                }
            }
            else {
                std::vector<int> indices(n);
                std::iota(indices.begin(), indices.end(), 0);
                std::shuffle(indices.begin(), indices.end(), *rng);
                for (int c = 0; c < k; ++c) {
                    int idx = indices[c % n];
                    std::copy(X.Col(idx), X.Col(idx) + d, centers.Col(c));
                }
            }

            for (int c = 0; c < k; ++c) {
                float* center = centers.Col(c);
                float norm = 0.0f;
#pragma omp simd reduction(+:norm)
                for (int r = 0; r < d; ++r) {
                    norm += center[r] * center[r];
                }
                norm = std::sqrt(std::max(norm, kEps));
                if (norm > kEps) {
#pragma omp simd
                    for (int r = 0; r < d; ++r) {
                        center[r] /= norm;
                    }
                }
            }
            return centers;
        }

        void AssignSamplesParallel(const ColMajorMatrix<float>& X,
                                   const ColMajorMatrix<float>& centers,
                                   std::vector<int>* assignments,
                                   std::vector<float>* dot_values) {
            // Julia reference: assign_samples_parallel! (fast_spkmeans.jl:218-248).
            const int d = X.rows;
            const int n = X.cols;
            const int k = centers.cols;
            assignments->assign(n, 0);
            dot_values->assign(n, 0.0f);

#pragma omp parallel for default(none) shared(X, centers, assignments, dot_values) firstprivate(d, n, k, kEps) schedule(static)
            for (int i = 0; i < n; ++i) {
                const float* xi = X.Col(i);
                int best = 0;
                float best_dot = -std::numeric_limits<float>::infinity();
                for (int c = 0; c < k; ++c) {
                    const float* center = centers.Col(c);
                    float dot = 0.0f;
#pragma omp simd reduction(+:dot)
                    for (int r = 0; r < d; ++r) {
                        dot += xi[r] * center[r];
                    }
                    if (dot > best_dot) {
                        best_dot = dot;
                        best = c;
                    }
                }
                (*assignments)[i] = best;
                (*dot_values)[i] = best_dot;
            }
        }

        void AssignSamplesGpu(const ColMajorMatrix<float>& X,
                              const ColMajorMatrix<float>& centers,
                              StreamKernelProvider* stream_kernels,
                              bool profile_timing,
                              KmeansTiming* timing,
                              std::vector<int>* assignments,
                              std::vector<float>* dot_values) {
            if (!assignments || !dot_values) {
                throw std::runtime_error("AssignSamplesGpu: output is null.");
            }

            if (!stream_kernels || !stream_kernels->IsGpu()) {
                if (profile_timing && timing) {
                    Timer t;
                    AssignSamplesParallel(X, centers, assignments, dot_values);
                    timing->assign_cpu_s += t.ElapsedSeconds();
                }
                else {
                    AssignSamplesParallel(X, centers, assignments, dot_values);
                }
                return;
            }

            const int d = X.rows;
            const int n = X.cols;
            const int k = centers.cols;

            // Avoid allocating the full k*n score matrix on GPU when n is very large (e.g. 1e6+).
            // We tile over columns of X: scores tile is k*tile_n floats.
            // Empirically, for large k (e.g. 4096) smaller score tiles can outperform very large tiles due to
            // cuBLAS heuristic choices and better shape alignment. Use a moderate fixed budget and align cols.
            // NOTE: We intentionally do NOT cap tile_n to 65535: for modern CUDA devices, grid.x can be much larger,
            // and small-K cases (e.g., K=256) benefit greatly from larger tiles (fewer GEMM launches).
            constexpr std::size_t kScoreBudgetBytes = 256ull * 1024ull * 1024ull; // 256 MiB
            int tile_n = static_cast<int>(kScoreBudgetBytes / (sizeof(float) * static_cast<std::size_t>(
                std::max(1, k))));
            tile_n = std::max(1, tile_n);
            tile_n = std::min(tile_n, n);
            // Prefer a warp-friendly column count (helps GEMM/argmax throughput, reduces tail effects).
            if (tile_n > 256) tile_n = (tile_n / 256) * 256;
            tile_n = std::max(1, tile_n);

            assignments->assign(static_cast<std::size_t>(n), 0);
            dot_values->assign(static_cast<std::size_t>(n), 0.0f);

            std::vector<int> idx_tile;
            std::vector<float> dot_tile;

            // Best-effort: cache the full X matrix on device once, then tile over device pointers.
            // This avoids repeated H2D inside GemmDeviceHostPtrB, which can dominate for large N.
            DeviceMatF32View Xdev;
            bool have_Xdev = false;
            try {
                have_Xdev = stream_kernels->EnsureDeviceF32(X, &Xdev);
            }
            catch (...) {
                have_Xdev = false;
                Xdev = {};
            }
            // NOTE: `KmeansTiming.used_device_xnorm_cache` is reserved for the streaming k-means
            // path where we may cache the *normalized* X on device (`KmeansTryCacheXnormOnDevice`).
            // Do not set it here: this helper is used by both in-memory and streaming codepaths.

            // Streaming-friendly implementation: tile over X to bound GPU score matrix size.
            for (int i0 = 0; i0 < n; i0 += tile_n) {
                const int cols = std::min(tile_n, n - i0);
                DeviceMatF32View scores;
                if (profile_timing && timing) {
                    Timer t;
                    if (have_Xdev && Xdev.ptr) {
                        const float* dX =
                            Xdev.ptr + static_cast<std::size_t>(i0) * static_cast<std::size_t>(Xdev.ld);
                        stream_kernels->GemmDeviceDevicePtrB(/*transA=*/true, /*transB=*/false,
                                                                        /*alpha=*/1.0f,
                                                                        centers,
                                                                        dX,
                                                                        /*ldB=*/Xdev.ld,
                                                                        /*rowsB=*/d,
                                                                        /*colsB=*/cols,
                                                                        /*beta=*/0.0f,
                                                                        &scores);
                    }
                    else {
                        const float* xptr_host =
                            X.data.data() + static_cast<std::size_t>(i0) * static_cast<std::size_t>(d);
                        stream_kernels->GemmDeviceHostPtrB(/*transA=*/true, /*transB=*/false,
                                                                      /*alpha=*/1.0f,
                                                                      centers,
                                                                      xptr_host,
                                                                      /*ldB=*/d,
                                                                      /*rowsB=*/d,
                                                                      /*colsB=*/cols,
                                                                      /*beta=*/0.0f,
                                                                      &scores);
                    }
                    stream_kernels->SyncCompute();
                    timing->assign_gemm_s += t.ElapsedSeconds();

                    t.Reset();
                    stream_kernels->ArgmaxColsF32(scores, &idx_tile, &dot_tile);
                    timing->assign_argmax_s += t.ElapsedSeconds();
                }
                else {
                    if (have_Xdev && Xdev.ptr) {
                        const float* dX =
                            Xdev.ptr + static_cast<std::size_t>(i0) * static_cast<std::size_t>(Xdev.ld);
                        stream_kernels->GemmDeviceDevicePtrB(/*transA=*/true, /*transB=*/false,
                                                                        /*alpha=*/1.0f,
                                                                        centers,
                                                                        dX,
                                                                        /*ldB=*/Xdev.ld,
                                                                        /*rowsB=*/d,
                                                                        /*colsB=*/cols,
                                                                        /*beta=*/0.0f,
                                                                        &scores);
                    }
                    else {
                        const float* xptr_host =
                            X.data.data() + static_cast<std::size_t>(i0) * static_cast<std::size_t>(d);
                        stream_kernels->GemmDeviceHostPtrB(/*transA=*/true, /*transB=*/false,
                                                                      /*alpha=*/1.0f,
                                                                      centers,
                                                                      xptr_host,
                                                                      /*ldB=*/d,
                                                                      /*rowsB=*/d,
                                                                      /*colsB=*/cols,
                                                                      /*beta=*/0.0f,
                                                                      &scores);
                    }
                    stream_kernels->ArgmaxColsF32(scores, &idx_tile, &dot_tile);
                }

                std::copy_n(idx_tile.begin(), cols,
                            assignments->begin() + static_cast<std::ptrdiff_t>(i0));
                std::copy_n(dot_tile.begin(), cols,
                            dot_values->begin() + static_cast<std::ptrdiff_t>(i0));
            }
        }

        float ComputeCost(const std::vector<float>& dot_values,
                          const std::vector<float>& weights) {
            // Julia reference: compute_cost_fast (fast_spkmeans.jl:255-272).
            const int n = static_cast<int>(dot_values.size());
            // Determinism: serial accumulation makes the stopping criterion and adaptive-weight
            // thresholds independent of OpenMP thread count.
            double total_cost = 0.0;
            double total_weight = 0.0;
            for (int i = 0; i < n; ++i) {
                const auto cost = static_cast<double>(1.0f - dot_values[i]);
                const auto w = static_cast<double>(weights[i]);
                total_cost += w * cost;
                total_weight += w;
            }
            return total_weight > 0.0 ? static_cast<float>(total_cost / total_weight) : 0.0f;
        }

        float ComputeQuantile(std::vector<float>* values, float q) {
            if (!values || values->empty()) {
                return 0.0f;
            }
            q = std::min(1.0f, std::max(0.0f, q));
            std::size_t n = values->size();
            auto idx = static_cast<std::size_t>(q * static_cast<float>(n - 1));
            std::nth_element(values->begin(), values->begin() + idx, values->end());
            return (*values)[idx];
        }


        void WriteConstantF32VectorOrThrow(const std::string& path, std::size_t n, float value) {
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            if (!out) {
                throw std::runtime_error("Failed to open for write: " + path);
            }
            constexpr std::size_t kChunk = 1u << 20; // 1M floats = 4 MiB
            std::vector<float> buf(std::min(kChunk, n), value);
            std::size_t left = n;
            while (left > 0) {
                const std::size_t take = std::min(left, buf.size());
                out.write(reinterpret_cast<const char*>(buf.data()),
                          static_cast<std::streamsize>(sizeof(float) * take));
                if (!out) {
                    throw std::runtime_error("Failed to write: " + path);
                }
                left -= take;
            }
        }

        template <typename T>
        void ReadVectorExactOrThrow(std::istream& in, std::vector<T>* v, std::size_t n) {
            v->resize(n);
            in.read(reinterpret_cast<char*>(v->data()),
                    static_cast<std::streamsize>(sizeof(T) * n));
            // NOTE: reaching EOF exactly at the end of the requested read may set `eofbit`
            // without indicating a short read. We only treat a short read as an error.
            if (in.gcount() != static_cast<std::streamsize>(sizeof(T) * n)) {
                throw std::runtime_error("Failed to read exact bytes from stream.");
            }
        }

        ColMajorMatrix<float> BuildNormalizedSample(const ColMajorMatrix<float>& X,
                                                    int sample_n,
                                                    std::mt19937* rng) {
            const int d = X.rows;
            const int n = X.cols;
            sample_n = std::min(n, std::max(1, sample_n));
            ColMajorMatrix<float> sample(d, sample_n);
            std::uniform_int_distribution<int> uni(0, std::max(0, n - 1));
            for (int i = 0; i < sample_n; ++i) {
                const int idx = uni(*rng);
                std::copy(X.Col(idx), X.Col(idx) + d, sample.Col(i));
            }
            return NormalizeColumns(sample);
        }

        inline bool AnnealingEnabled(const KmeansConfig& cfg) {
            return cfg.annealing_factor > 0.0f && cfg.warmup_iters < cfg.max_iters;
        }

        void UpdateCenters(const ColMajorMatrix<float>& X,
                           const std::vector<int>& assignments,
                           const std::vector<float>& weights,
                           std::mt19937* rng,
                           ColMajorMatrix<float>* centers) {
            // Julia reference: update_centers_fast! (fast_spkmeans.jl:278-364).
            const int d = X.rows;
            const int n = X.cols;
            const int k = centers->cols;
            // Memory/perf: avoid `nthreads * (k*d)` accumulators when k is large (e.g. 4096/65536).
            // Instead, bucket points by assignment, then reduce per-cluster with thread-local `d` buffers.
            struct UpdateCentersScratch
            {
                std::vector<int> offsets;
                std::vector<int> order;
                std::vector<int> cursor;
                std::vector<int> fallback;
            };
            static thread_local UpdateCentersScratch scratch;
            scratch.offsets.resize(static_cast<std::size_t>(k) + 1);
            std::fill(scratch.offsets.begin(), scratch.offsets.end(), 0);

            std::vector<int>& offsets = scratch.offsets;
            for (int i = 0; i < n; ++i) {
                const int c = assignments[static_cast<std::size_t>(i)];
                offsets[static_cast<std::size_t>(c) + 1] += 1;
            }
            for (int c = 0; c < k; ++c) {
                offsets[static_cast<std::size_t>(c) + 1] += offsets[static_cast<std::size_t>(c)];
            }
            scratch.order.resize(static_cast<std::size_t>(n));
            std::vector<int>& order = scratch.order;
            scratch.cursor = offsets; // same size; copies offsets values
            std::vector<int>& cursor = scratch.cursor;
            for (int i = 0; i < n; ++i) {
                const int c = assignments[static_cast<std::size_t>(i)];
                const int pos = cursor[static_cast<std::size_t>(c)]++;
                order[static_cast<std::size_t>(pos)] = i;
            }

            // Deterministic fallback samples for empty clusters.
            scratch.fallback.resize(static_cast<std::size_t>(k));
            std::vector<int>& fallback = scratch.fallback;
            {
                std::uniform_int_distribution<int> uni(0, std::max(0, n - 1));
                for (int c = 0; c < k; ++c) {
                    fallback[static_cast<std::size_t>(c)] = uni(*rng);
                }
            }

#pragma omp parallel default(none) shared(centers, offsets, fallback, X, order, weights) firstprivate(d, k, kEps)
            {
                // Performance: float accumulation is substantially faster and reduces memory bandwidth.
                // Minor numeric drift is acceptable under GPU/parallel execution.
                std::vector<float> sum(static_cast<std::size_t>(d), 0.0f);

#pragma omp for schedule(static)
                for (int c = 0; c < k; ++c) {
                    float* center = centers->Col(c);
                    const int begin = offsets[static_cast<std::size_t>(c)];
                    const int end = offsets[static_cast<std::size_t>(c) + 1];
                    if (begin == end) {
                        const int idx = fallback[static_cast<std::size_t>(c)];
                        std::copy(X.Col(idx), X.Col(idx) + d, center);
                        continue;
                    }

                    std::fill(sum.begin(), sum.end(), 0.0f);
                    float wsum = 0.0f;
                    for (int p = begin; p < end; ++p) {
                        const int i = order[static_cast<std::size_t>(p)];
                        const float w = weights[static_cast<std::size_t>(i)];
                        wsum += w;
                        const float* xi = X.Col(i);
#pragma omp simd
                        for (int r = 0; r < d; ++r) sum[static_cast<std::size_t>(r)] += w * xi[r];
                    }

                    if (wsum <= kEps) {
                        const int idx = fallback[static_cast<std::size_t>(c)];
                        std::copy(X.Col(idx), X.Col(idx) + d, center);
                        continue;
                    }

                    float norm = 0.0f;
                    for (int r = 0; r < d; ++r) {
                        const float v = sum[static_cast<std::size_t>(r)] / wsum;
                        center[r] = v;
                        norm += v * v;
                    }
                    norm = std::sqrt(std::max(norm, kEps));
                    const float inv = 1.0f / norm;
#pragma omp simd
                    for (int r = 0; r < d; ++r) {
                        center[r] *= inv;
                    }
                }
            }
        }
    } // namespace

    ColMajorMatrix<float> SphericalKmeansCentersOnlyStreamingOnePass(const ColMajorMatrix<float>& X,
                                                                     int k,
                                                                     const KmeansConfig& cfg,
                                                                     int block_cols,
                                                                     std::mt19937* rng,
                                                                     StreamKernelProvider* stream_kernels,
                                                                     bool profile_timing,
                                                                     KmeansTiming* timing) {
        const int d = X.rows;
        const int n = X.cols;
        if (d <= 0 || n <= 0 || k <= 0) {
            return {d, std::max(0, k)};
        }
        block_cols = std::min(n, std::max(1, block_cols));

        // NOTE: This one-pass variant assumes fixed weights (no adaptive annealing update).
        // A two-pass streaming algorithm is required once annealing/quantile weights are enabled.

        Timer t_total;

        // Init centers from a normalized random sample.
        ColMajorMatrix<float> centers;
        {
            Timer t_init;
            ColMajorMatrix<float> sample_norm;
            {
                Timer t_sample;
                sample_norm = BuildNormalizedSample(X, cfg.init_samples, rng);
                if (profile_timing && timing) timing->init_sample_s += t_sample.ElapsedSeconds();
            }
            {
                Timer t_pick;
                centers = InitializeCentersFast(sample_norm, k, cfg, rng);
                if (profile_timing && timing) timing->init_pick_s += t_pick.ElapsedSeconds();
            }
            if (profile_timing && timing) timing->init_centers_s += t_init.ElapsedSeconds();
        }

        // Reusable block buffer (normalized).
        ColMajorMatrix<float> Xblk_norm(d, block_cols);
        std::vector<int> assign_blk;
        std::vector<float> dot_blk;

        // Global accumulators per-iteration.
        ColMajorMatrix<float> sum_x(d, k);
        std::vector<float> sum_w(static_cast<std::size_t>(k), 0.0f);

        // Fallback samples for empty clusters.
        std::vector<int> fallback(static_cast<std::size_t>(k), 0);
        {
            std::uniform_int_distribution<int> uni(0, std::max(0, n - 1));
            for (int c = 0; c < k; ++c) fallback[static_cast<std::size_t>(c)] = uni(*rng);
        }

        double prev_cost = std::numeric_limits<double>::infinity();
        for (int iter = 1; iter <= cfg.max_iters; ++iter) {
            Timer t_iter;
            std::fill(sum_x.data.begin(), sum_x.data.end(), 0.0f);
            std::fill(sum_w.begin(), sum_w.end(), 0.0f);

            double total_cost = 0.0;
            std::int64_t total_n = 0;

            for (int i0 = 0; i0 < n; i0 += block_cols) {
                const int cols = std::min(block_cols, n - i0);
                if (cols != Xblk_norm.cols) {
                    Xblk_norm = ColMajorMatrix<float>(d, cols);
                }

                // Normalize this block.
                Timer t_norm;
#pragma omp parallel for default(none) shared(X, Xblk_norm) firstprivate(i0, cols, d, kEps) schedule(static)
                for (int j = 0; j < cols; ++j) {
                    const float* xi = X.Col(i0 + j);
                    float* dst = Xblk_norm.Col(j);
                    float norm = 0.0f;
#pragma omp simd reduction(+:norm)
                    for (int r = 0; r < d; ++r) {
                        norm += xi[r] * xi[r];
                    }
                    norm = std::sqrt(std::max(norm, kEps));
                    const float inv = (norm > kEps) ? (1.0f / norm) : 1.0f;
#pragma omp simd
                    for (int r = 0; r < d; ++r) {
                        dst[r] = xi[r] * inv;
                    }
                }
                if (profile_timing && timing) timing->normalize_s += t_norm.ElapsedSeconds();

                // Assignment (GPU GEMM + argmax).
                AssignSamplesGpu(Xblk_norm, centers, stream_kernels, profile_timing, timing,
                                 &assign_blk, &dot_blk);

                // Accumulate cost.
#pragma omp simd reduction(+:total_cost)
                for (int j = 0; j < cols; ++j) {
                    total_cost += static_cast<double>(1.0f - dot_blk[static_cast<std::size_t>(j)]);
                }
                total_n += cols;

                // Bucket within this block (per-cluster contiguous ranges).
                std::vector<int> offsets(static_cast<std::size_t>(k) + 1, 0);
                for (int j = 0; j < cols; ++j) {
                    const int c = assign_blk[static_cast<std::size_t>(j)];
                    ++offsets[static_cast<std::size_t>(c) + 1];
                }
                for (int c = 0; c < k; ++c) {
                    offsets[static_cast<std::size_t>(c) + 1] += offsets[static_cast<std::size_t>(c)];
                }
                std::vector<int> order(static_cast<std::size_t>(cols), 0);
                std::vector<int> cursor = offsets;
                for (int j = 0; j < cols; ++j) {
                    const int c = assign_blk[static_cast<std::size_t>(j)];
                    const int pos = cursor[static_cast<std::size_t>(c)]++;
                    order[static_cast<std::size_t>(pos)] = j;
                }

                // Reduce per cluster, then add into global sum_x/sum_w (no atomics needed).
#pragma omp parallel default(none) shared(offsets, order, Xblk_norm, sum_x, sum_w) firstprivate(d, k)
                {
                    std::vector<float> local_sum(static_cast<std::size_t>(d), 0.0f);
#pragma omp for schedule(static)
                    for (int c = 0; c < k; ++c) {
                        const int begin = offsets[static_cast<std::size_t>(c)];
                        const int end = offsets[static_cast<std::size_t>(c) + 1];
                        if (begin == end) continue;

                        std::fill(local_sum.begin(), local_sum.end(), 0.0f);
                        for (int p = begin; p < end; ++p) {
                            const int j = order[static_cast<std::size_t>(p)];
                            const float* xj = Xblk_norm.Col(j);
#pragma omp simd
                            for (int r = 0; r < d; ++r) {
                                local_sum[static_cast<std::size_t>(r)] += xj[r];
                            }
                        }
                        float* dst = sum_x.Col(c);
#pragma omp simd
                        for (int r = 0; r < d; ++r) {
                            dst[r] += local_sum[static_cast<std::size_t>(r)];
                        }
                        sum_w[static_cast<std::size_t>(c)] += static_cast<float>(end - begin);
                    }
                }
            }

            // Update centers from sums.
            Timer t_upd;
#pragma omp parallel for default(none) shared(centers, sum_w, fallback, X, sum_x) firstprivate(k, d, kEps) schedule(static)
            for (int c = 0; c < k; ++c) {
                float* center = centers.Col(c);
                const float w = sum_w[static_cast<std::size_t>(c)];
                if (w <= kEps) {
                    // Empty cluster: reseed from a random normalized sample.
                    const int idx = fallback[static_cast<std::size_t>(c)];
                    const float* xi = X.Col(idx);
                    float norm = 0.0f;
#pragma omp simd reduction(+:norm)
                    for (int r = 0; r < d; ++r) norm += xi[r] * xi[r];
                    norm = std::sqrt(std::max(norm, kEps));
                    const float inv = (norm > kEps) ? (1.0f / norm) : 1.0f;
#pragma omp simd
                    for (int r = 0; r < d; ++r) center[r] = xi[r] * inv;
                    continue;
                }
                const float inv_w = 1.0f / w;
                const float* sx = sum_x.Col(c);
                float norm = 0.0f;
                for (int r = 0; r < d; ++r) {
                    const float v = sx[r] * inv_w;
                    center[r] = v;
                    norm += v * v;
                }
                norm = std::sqrt(std::max(norm, kEps));
                const float inv = 1.0f / norm;
#pragma omp simd
                for (int r = 0; r < d; ++r) center[r] *= inv;
            }
            if (profile_timing && timing) timing->update_s += t_upd.ElapsedSeconds();

            const double current_cost = (total_n > 0) ? (total_cost / static_cast<double>(total_n)) : 0.0;
            if (cfg.progress_every > 0 &&
                (iter == 1 || (iter % cfg.progress_every) == 0 || iter == cfg.max_iters)) {
                LogInfo("[kmeans] iter=" + std::to_string(iter) + "/" + std::to_string(cfg.max_iters) +
                    " cost=" + std::to_string(current_cost) +
                    " n=" + std::to_string(total_n) +
                    " dt=" + std::to_string(t_iter.ElapsedSeconds()) + "s");
            }
            if (std::abs(prev_cost - current_cost) < static_cast<double>(cfg.tol)) {
                break;
            }
            prev_cost = current_cost;
        }

        if (profile_timing && timing) timing->total_s += t_total.ElapsedSeconds();
        return centers;
    }

    namespace
    {
        struct RvqResidualHook
        {
            const RvqInitCodesInMemory* codes = nullptr;
            const std::vector<ColMajorMatrix<float>>* codebooks = nullptr;
            int upto_layer = 0;
            // Prompt-10: optionally emit this layer's codes during the last-iteration update scan.
            RvqInitCodesInMemory* emit_codes = nullptr;
            int emit_layer = -1;
        };

        thread_local const RvqResidualHook* g_rvq_hook = nullptr;

        struct ScopedRvqResidualHook
        {
            const RvqResidualHook* prev = nullptr;

            explicit ScopedRvqResidualHook(const RvqResidualHook* hook) {
                prev = g_rvq_hook;
                g_rvq_hook = hook;
            }

            ~ScopedRvqResidualHook() { g_rvq_hook = prev; }
        };

        ColMajorMatrix<float> BuildNormalizedSampleFromReader(io::IColBlockReader* reader,
                                                              int sample_n,
                                                              int block_cols,
                                                              std::mt19937* rng,
                                                              std::string* err) {
            if (!reader) {
                if (err) *err = "BuildNormalizedSampleFromReader: reader is null.";
                return {};
            }
            const int d = reader->d();
            const std::int64_t n64 = reader->n();
            if (d <= 0 || n64 <= 0) {
                return {d, 0};
            }
            sample_n = std::min<std::int64_t>(std::max(1, sample_n), n64);
            block_cols = std::max(1, block_cols);

            // Performance (Prompt-1A/1B):
            // - Avoid reservoir sampling over the full dataset. `sample_n` is treated as the init subset size.
            // - Use ReadNextInto + reusable buffers and block-parallel normalize to reduce CPU overhead.
            const bool want_residual = (g_rvq_hook != nullptr && g_rvq_hook->upto_layer > 0);
            std::vector<std::int64_t> sample_idx;
            if (want_residual) {
                sample_idx.resize(static_cast<std::size_t>(sample_n));
            }

            ColMajorMatrix<float> sample_norm(d, sample_n);
            // Pick a random contiguous window [start, start+sample_n) for the init subset (single seek).
            std::int64_t start = 0;
            if (rng && n64 > static_cast<std::int64_t>(sample_n)) {
                std::uniform_int_distribution<std::int64_t> dist(0, n64 - static_cast<std::int64_t>(sample_n));
                start = dist(*rng);
            }
            if (start > 0) {
                std::string seek_err;
                if (!reader->Seek(start, &seek_err)) {
                    // Best-effort: if seek is unsupported, fall back to the first window (still fast).
                    start = 0;
                }
            }
            if (start == 0) {
                if (!reader->Reset(err)) {
                    return {};
                }
            }

            const io::ColBlockDType dt = reader->dtype();
            const bool src_u8 = (dt == io::ColBlockDType::kU8);
            const bool src_f32 = (dt == io::ColBlockDType::kF32);
            if (!src_u8 && !src_f32) {
                if (err) *err = "BuildNormalizedSampleFromReader: unsupported reader dtype.";
                return {};
            }

            const int buf_cols = std::min(block_cols, sample_n);
            std::vector<std::uint8_t> buf_u8;
            std::vector<float> buf_f32;
            if (src_u8) {
                buf_u8.resize(static_cast<std::size_t>(d) * static_cast<std::size_t>(buf_cols));
            }
            else {
                buf_f32.resize(static_cast<std::size_t>(d) * static_cast<std::size_t>(buf_cols));
            }

            int filled = 0;
            while (filled < sample_n) {
                std::int64_t col0 = 0;
                int cols = 0;
                const bool ok = reader->ReadNextInto(
                    buf_cols,
                    src_u8 ? static_cast<void*>(buf_u8.data()) : static_cast<void*>(buf_f32.data()),
                    src_u8 ? buf_u8.size() : (buf_f32.size() * sizeof(float)),
                    &col0,
                    &cols,
                    err);
                if (!ok) {
                    // Fallback: reader does not support ReadNextInto.
                    io::ColBlock blk;
                    if (!reader->ReadNext(buf_cols, &blk, err)) {
                        return {};
                    }
                    cols = (blk.dtype == io::ColBlockDType::kU8) ? blk.X_u8.cols : blk.X_f32.cols;
                    if (cols <= 0) break;
                    col0 = blk.col0;
                    const int take = std::min(cols, sample_n - filled);
#pragma omp parallel for default(none) shared(sample_norm, blk, sample_idx) firstprivate(filled, take, d, col0, want_residual, kEps) schedule(static)
                    for (int j = 0; j < take; ++j) {
                        float* dst = sample_norm.Col(filled + j);
                        float ss = 0.0f;
                        if (blk.dtype == io::ColBlockDType::kU8) {
                            const std::uint8_t* xj = blk.X_u8.Col(j);
#pragma omp simd reduction(+:ss)
                            for (int r = 0; r < d; ++r) {
                                const auto v = static_cast<float>(xj[r]);
                                dst[r] = v;
                                ss += v * v;
                            }
                        }
                        else {
                            const float* xj = blk.X_f32.Col(j);
#pragma omp simd reduction(+:ss)
                            for (int r = 0; r < d; ++r) {
                                const float v = xj[r];
                                dst[r] = v;
                                ss += v * v;
                            }
                        }
                        const float inv = 1.0f / std::sqrt(std::max(ss, kEps));
#pragma omp simd
                        for (int r = 0; r < d; ++r) dst[r] *= inv;
                        if (want_residual) {
                            sample_idx[static_cast<std::size_t>(filled + j)] = col0 + j;
                        }
                    }
                    filled += take;
                    continue;
                }
                if (cols <= 0) break;
                const int take = std::min(cols, sample_n - filled);
#pragma omp parallel for default(none) shared(sample_norm, buf_u8, buf_f32, sample_idx) firstprivate(filled, take, d, col0, want_residual, src_u8, kEps) schedule(static)
                for (int j = 0; j < take; ++j) {
                    float* dst = sample_norm.Col(filled + j);
                    float ss = 0.0f;
                    if (src_u8) {
                        const std::uint8_t* xj = buf_u8.data() + static_cast<std::size_t>(j) * static_cast<std::size_t>(
                            d);
#pragma omp simd reduction(+:ss)
                        for (int r = 0; r < d; ++r) {
                            const auto v = static_cast<float>(xj[r]);
                            dst[r] = v;
                            ss += v * v;
                        }
                    }
                    else {
                        const float* xj = buf_f32.data() + static_cast<std::size_t>(j) * static_cast<std::size_t>(d);
#pragma omp simd reduction(+:ss)
                        for (int r = 0; r < d; ++r) {
                            const float v = xj[r];
                            dst[r] = v;
                            ss += v * v;
                        }
                    }
                    const float inv = 1.0f / std::sqrt(std::max(ss, kEps));
#pragma omp simd
                    for (int r = 0; r < d; ++r) dst[r] *= inv;
                    if (want_residual) {
                        sample_idx[static_cast<std::size_t>(filled + j)] = col0 + j;
                    }
                }
                filled += take;
            }

            // Reset for subsequent full scans.
            if (!reader->Reset(err)) {
                return {};
            }
            if (filled != sample_n) {
                if (err) *err = "BuildNormalizedSampleFromReader: failed to fill init sample (unexpected EOF).";
                return {};
            }
            if (!want_residual) {
                return sample_norm;
            }

            // NOTE: `g_rvq_hook` is thread_local. Capture the hook pointer from the caller thread and
            // use that shared pointer inside OpenMP regions, otherwise worker threads would see a null
            // hook and crash.
            const auto* hook = g_rvq_hook;
            if (!hook || !hook->codes || !hook->codebooks) {
                if (err) *err = "BuildNormalizedSampleFromReader: rvq hook missing codes/codebooks.";
                return {};
            }
            const int upto = hook->upto_layer;
            if (upto > static_cast<int>(hook->codebooks->size())) {
                if (err) *err = "BuildNormalizedSampleFromReader: codebooks too small for upto_layer.";
                return {};
            }

            // Apply residual projections on CPU for the init sample only (one-time cost per layer).
            // This is sufficient because the performance-critical full scans use GPU projection.
#pragma omp parallel for default(none) shared(sample_norm, sample_idx, hook) firstprivate(d, upto) schedule(static)
            for (int j = 0; j < sample_norm.cols; ++j) {
                float* x = sample_norm.Col(j);
                const std::int64_t gi = sample_idx[static_cast<std::size_t>(j)];
                for (int l = 0; l < upto; ++l) {
                    const std::uint32_t code_u32 = hook->codes->CodeAt(l, gi);
                    const int code = static_cast<int>(code_u32);
                    const auto& C = (*hook->codebooks)[static_cast<std::size_t>(l)];
                    if (code < 0 || code >= C.cols) continue;
                    const float* center = C.Col(code);
                    float dot = 0.0f;
#pragma omp simd reduction(+:dot)
                    for (int r = 0; r < d; ++r) dot += x[r] * center[r];
#pragma omp simd
                    for (int r = 0; r < d; ++r) x[r] -= dot * center[r];
                }
            }
            NormalizeColumnsInPlace(&sample_norm);
            return sample_norm;
        }

        DeviceMatF32View UploadAndNormalizeBlock(CudaStreamKernels* cuda_kernels,
                                                 const io::ColBlock& blk,
                                                 bool stage_to_pinned,
                                                 bool profile_timing,
                                                 KmeansTiming* timing) {
            const int d = (blk.dtype == io::ColBlockDType::kU8) ? blk.X_u8.rows : blk.X_f32.rows;
            const int cols = (blk.dtype == io::ColBlockDType::kU8) ? blk.X_u8.cols : blk.X_f32.cols;
            if (!cuda_kernels || d <= 0 || cols <= 0) {
                return DeviceMatF32View{};
            }
            if (profile_timing && timing) {
                Timer t_norm;
                double stage_s = 0.0;
                double h2d_s = 0.0;
                double ker_s = 0.0;
                DeviceMatF32View Xd;
                if (blk.dtype == io::ColBlockDType::kU8) {
                    Xd = cuda_kernels->KmeansUploadU8AndNormalizeHostPtrBAsync(
                        /*slot=*/0,
                                 blk.X_u8.data.data(), /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols,
                                 stage_to_pinned,
                                 &stage_s, &h2d_s, &ker_s);
                    if (g_rvq_hook && g_rvq_hook->upto_layer > 0) {
                        std::vector<const void*> codes_ptr(static_cast<std::size_t>(g_rvq_hook->upto_layer), nullptr);
                        std::vector<int> code_bytes(static_cast<std::size_t>(g_rvq_hook->upto_layer), 0);
                        for (int l = 0; l < g_rvq_hook->upto_layer; ++l) {
                            std::string local_err;
                            const RvqCodeSpan span = g_rvq_hook->codes->SpanBlock(l, blk.col0, cols, &local_err);
                            if (!local_err.empty()) {
                                throw std::runtime_error(
                                    "UploadAndNormalizeBlock(rvq): SpanBlock failed: " + local_err);
                            }
                            codes_ptr[static_cast<std::size_t>(l)] = span.ptr;
                            code_bytes[static_cast<std::size_t>(l)] = RvqCodeBytes(span.dtype);
                        }
                        cuda_kernels->RvqStageCodesBlockInCopyStreamAfterUpload(
                            /*slot=*/0, cols, g_rvq_hook->upto_layer, codes_ptr.data(), code_bytes.data());
                    }
                    // Insert dependency for subsequent compute kernels on the compute stream.
                    cuda_kernels->KmeansComputeWaitForUpload(0);
                    if (g_rvq_hook && g_rvq_hook->upto_layer > 0) {
                        cuda_kernels->RvqProjectXnormBlockOnComputeStream(/*slot=*/0, Xd, cols, g_rvq_hook->upto_layer);
                    }
                }
                else {
                    Xd = cuda_kernels->KmeansUploadAndNormalizeHostPtrBTimed(
                        blk.X_f32.data.data(), /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols,
                        stage_to_pinned,
                        &stage_s, &h2d_s, &ker_s);
                }
                timing->normalize_stage_s += stage_s;
                timing->normalize_h2d_s += h2d_s;
                timing->normalize_kernel_s += ker_s;
                timing->normalize_s += t_norm.ElapsedSeconds();
                return Xd;
            }
            if (blk.dtype == io::ColBlockDType::kU8) {
                DeviceMatF32View Xd = cuda_kernels->KmeansUploadU8AndNormalizeHostPtrBAsync(
                    /*slot=*/0,
                             blk.X_u8.data.data(), /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols,
                             stage_to_pinned,
                             /*stage_s=*/nullptr,
                             /*h2d_s=*/nullptr,
                             /*kernel_s=*/nullptr);
                if (g_rvq_hook && g_rvq_hook->upto_layer > 0) {
                    std::vector<const void*> codes_ptr(static_cast<std::size_t>(g_rvq_hook->upto_layer), nullptr);
                    std::vector<int> code_bytes(static_cast<std::size_t>(g_rvq_hook->upto_layer), 0);
                    for (int l = 0; l < g_rvq_hook->upto_layer; ++l) {
                        std::string local_err;
                        const RvqCodeSpan span = g_rvq_hook->codes->SpanBlock(l, blk.col0, cols, &local_err);
                        if (!local_err.empty()) {
                            throw std::runtime_error("UploadAndNormalizeBlock(rvq): SpanBlock failed: " + local_err);
                        }
                        codes_ptr[static_cast<std::size_t>(l)] = span.ptr;
                        code_bytes[static_cast<std::size_t>(l)] = RvqCodeBytes(span.dtype);
                    }
                    cuda_kernels->RvqStageCodesBlockInCopyStreamAfterUpload(
                        /*slot=*/0, cols, g_rvq_hook->upto_layer, codes_ptr.data(), code_bytes.data());
                }
                cuda_kernels->KmeansComputeWaitForUpload(0);
                if (g_rvq_hook && g_rvq_hook->upto_layer > 0) {
                    cuda_kernels->RvqProjectXnormBlockOnComputeStream(/*slot=*/0, Xd, cols, g_rvq_hook->upto_layer);
                }
                return Xd;
            }
            return cuda_kernels->KmeansUploadAndNormalizeHostPtrBTimed(
                blk.X_f32.data.data(), /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols,
                stage_to_pinned,
                /*stage_s=*/nullptr,
                /*h2d_s=*/nullptr,
                /*kernel_s=*/nullptr);
        }

        void AssignBlockGpuTiled(const DeviceMatF32View& Xd,
                                 const ColMajorMatrix<float>& centers,
                                 int k,
                                 StreamKernelProvider* stream_kernels,
                                 CudaStreamKernels* cuda_kernels,
                                 bool profile_timing,
                                 KmeansTiming* timing,
                                 bool need_host_argmax,
                                 std::vector<int>* assign_blk,
                                 std::vector<float>* dot_blk) {
            const int cols = Xd.cols;
            if (!cuda_kernels || !stream_kernels || cols <= 0) {
                return;
            }

            constexpr std::size_t kScoreBudgetBytes = 256ull * 1024ull * 1024ull; // 256 MiB
            int tile_n = static_cast<int>(kScoreBudgetBytes /
                (sizeof(float) * static_cast<std::size_t>(std::max(1, k))));
            tile_n = std::max(1, tile_n);
            tile_n = std::min(tile_n, cols);
            if (tile_n > 256) tile_n = (tile_n / 256) * 256;
            tile_n = std::max(1, tile_n);

            cuda_kernels->KmeansEnsureBlockArgmax(cols);
            DeviceVecI32View d_assign_tile;
            DeviceVecF32View d_best_tile;
            for (int j0 = 0; j0 < cols; j0 += tile_n) {
                const int tcols = std::min(tile_n, cols - j0);
                const float* dX = Xd.ptr + static_cast<std::size_t>(j0) * static_cast<std::size_t>(Xd.ld);

                DeviceMatF32View scores;
                Timer t_gemm;
                stream_kernels->GemmDeviceDevicePtrB(/*transA=*/true, /*transB=*/false,
                                                                /*alpha=*/1.0f,
                                                                centers,
                                                                dX,
                                                                /*ldB=*/Xd.ld,
                                                                /*rowsB=*/Xd.rows,
                                                                /*colsB=*/tcols,
                                                                /*beta=*/0.0f,
                                                                &scores);
                if (profile_timing && timing) {
                    stream_kernels->SyncCompute();
                    timing->assign_gemm_s += t_gemm.ElapsedSeconds();
                }

                Timer t_arg;
                stream_kernels->ArgmaxColsF32Device(scores, &d_assign_tile, &d_best_tile);
                cuda_kernels->KmeansCopyTileArgmaxToBlock(d_assign_tile, d_best_tile, j0);
                if (profile_timing && timing) {
                    stream_kernels->SyncCompute();
                    timing->assign_argmax_s += t_arg.ElapsedSeconds();
                }
            }
            if (need_host_argmax) {
                cuda_kernels->KmeansDownloadBlockArgmax(assign_blk, dot_blk);
            }
        }

        void AccumulateBlockCpu(const ColMajorMatrix<float>& Xnorm,
                                const std::vector<int>& assign_blk,
                                const std::vector<float>* w_blk,
                                ColMajorMatrix<float>* sum_x,
                                std::vector<float>* sum_w,
                                std::vector<int>* offsets,
                                std::vector<int>* cursor,
                                std::vector<int>* order) {
            if (!sum_x || !sum_w || !offsets || !cursor || !order) return;
            const int d = Xnorm.rows;
            const int cols = Xnorm.cols;
            const int k = sum_x->cols;
            if (cols <= 0 || d <= 0 || k <= 0) return;
            if (static_cast<int>(assign_blk.size()) != cols) return;
            if (w_blk && static_cast<int>(w_blk->size()) != cols) return;

            offsets->assign(static_cast<std::size_t>(k) + 1, 0);
            for (int j = 0; j < cols; ++j) {
                const int c = assign_blk[static_cast<std::size_t>(j)];
                (*offsets)[static_cast<std::size_t>(c) + 1] += 1;
            }
            for (int c = 0; c < k; ++c) {
                (*offsets)[static_cast<std::size_t>(c) + 1] += (*offsets)[static_cast<std::size_t>(c)];
            }

            order->resize(static_cast<std::size_t>(cols));
            *cursor = *offsets;
            for (int j = 0; j < cols; ++j) {
                const int c = assign_blk[static_cast<std::size_t>(j)];
                const int pos = (*cursor)[static_cast<std::size_t>(c)]++;
                (*order)[static_cast<std::size_t>(pos)] = j;
            }

#pragma omp parallel default(none) shared(Xnorm, w_blk, sum_x, sum_w, offsets, order) firstprivate(d, k)
            {
                std::vector<float> local_sum(static_cast<std::size_t>(d));
#pragma omp for schedule(static)
                for (int c = 0; c < k; ++c) {
                    const int begin = (*offsets)[static_cast<std::size_t>(c)];
                    const int end = (*offsets)[static_cast<std::size_t>(c) + 1];
                    if (begin == end) continue;

                    std::fill(local_sum.begin(), local_sum.end(), 0.0f);
                    float wsum = 0.0f;
                    for (int p = begin; p < end; ++p) {
                        const int j = (*order)[static_cast<std::size_t>(p)];
                        const float w = w_blk ? (*w_blk)[static_cast<std::size_t>(j)] : 1.0f;
                        wsum += w;
                        const float* xj = Xnorm.Col(j);
#pragma omp simd
                        for (int r = 0; r < d; ++r) {
                            local_sum[static_cast<std::size_t>(r)] += w * xj[r];
                        }
                    }
                    float* dst = sum_x->Col(c);
#pragma omp simd
                    for (int r = 0; r < d; ++r) {
                        dst[r] += local_sum[static_cast<std::size_t>(r)];
                    }
                    (*sum_w)[static_cast<std::size_t>(c)] += wsum;
                }
            }
        }

        inline std::string ToLowerAscii(std::string s) {
            for (char& ch : s) {
                const auto c = static_cast<unsigned char>(ch);
                if (c >= 'A' && c <= 'Z') ch = static_cast<char>(c - 'A' + 'a');
            }
            return s;
        }

        inline bool IsKmeansllInitMethod(const std::string& init_method) {
            const std::string m = ToLowerAscii(init_method);
            return (m == "kmeans||") || (m == "kmeansll") || (m == "kmeansll_gpu");
        }

        struct KmeansllSeedOut
        {
            ColMajorMatrix<float> centers; // d x K
            ColMajorMatrix<float> fallback_sample; // d x K (for empty cluster reseed)
        };

        DeviceMatF32View BuildKmeansllSampleOnDeviceFromReader(io::IColBlockReader* reader,
                                                               int sample_n,
                                                               int block_cols,
                                                               std::mt19937* rng,
                                                               CudaStreamKernels* cuda,
                                                               bool stage_blocks_to_pinned,
                                                               std::string* err) {
            if (!reader || !cuda) {
                if (err) *err = "BuildKmeansllSampleOnDeviceFromReader: null args.";
                return DeviceMatF32View{};
            }
            const int d = reader->d();
            const std::int64_t n64 = reader->n();
            if (d <= 0 || n64 <= 0) return DeviceMatF32View{};
            sample_n = static_cast<int>(std::min<std::int64_t>(std::max<std::int64_t>(1, sample_n), n64));
            block_cols = std::max(1, block_cols);

            // Pick a random contiguous window [start, start+sample_n) (single seek).
            std::int64_t start = 0;
            if (rng && n64 > static_cast<std::int64_t>(sample_n)) {
                std::uniform_int_distribution<std::int64_t> dist(0, n64 - static_cast<std::int64_t>(sample_n));
                start = dist(*rng);
            }
            if (start > 0) {
                std::string seek_err;
                if (!reader->Seek(start, &seek_err)) {
                    start = 0;
                }
            }
            if (start == 0) {
                if (!reader->Reset(err)) return DeviceMatF32View{};
            }

            const DeviceMatF32View dS = cuda->KmeansllEnsureSampleBuffer(d, sample_n);
            if (!dS.ptr || dS.cols != sample_n) {
                if (err) *err = "BuildKmeansllSampleOnDeviceFromReader: failed to allocate device sample buffer.";
                return DeviceMatF32View{};
            }

            const int slot = 0;
            int filled = 0;
            io::ColBlock blk;
            while (filled < sample_n) {
                const int want = std::min(block_cols, sample_n - filled);
                const io::ColBlockDType dt = reader->dtype();
                std::int64_t col0 = 0;
                int cols = 0;
                DeviceMatF32View Xd;

                bool used_direct = false;
                if (dt == io::ColBlockDType::kU8) {
                    std::uint8_t* pinned = cuda->KmeansGetPinnedStageBufferU8(slot, d, want);
                    const std::size_t bytes = sizeof(std::uint8_t) * static_cast<std::size_t>(d) * static_cast<
                        std::size_t>(want);
                    std::string local_err;
                    const bool ok = reader->ReadNextInto(want, pinned, bytes, &col0, &cols, &local_err);
                    if (!ok) {
                        // Fallback: non-direct reader.
                        if (!reader->ReadNext(want, &blk, err)) return DeviceMatF32View{};
                        cols = (blk.dtype == io::ColBlockDType::kU8) ? blk.X_u8.cols : blk.X_f32.cols;
                        if (cols <= 0) break;
                        col0 = blk.col0;
                        Xd = cuda->KmeansUploadU8AndNormalizeHostPtrBAsync(
                            slot,
                            blk.X_u8.data.data(),
                            /*ldB=*/blk.X_u8.rows,
                            /*rowsB=*/d,
                            /*colsB=*/cols,
                            /*stage_to_pinned=*/stage_blocks_to_pinned,
                            /*stage_s=*/nullptr,
                            /*h2d_s=*/nullptr,
                            /*kernel_s=*/nullptr);
                    }
                    else {
                        used_direct = true;
                        if (cols <= 0) break;
                        Xd = cuda->KmeansUploadU8AndNormalizePinnedAsync(
                            slot, pinned, /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols,
                            /*h2d_s=*/nullptr, /*kernel_s=*/nullptr);
                    }
                }
                else {
                    float* pinned = cuda->KmeansGetPinnedStageBuffer(slot, d, want);
                    const std::size_t bytes = sizeof(float) * static_cast<std::size_t>(d) * static_cast<std::size_t>(
                        want);
                    std::string local_err;
                    const bool ok = reader->ReadNextInto(want, pinned, bytes, &col0, &cols, &local_err);
                    if (!ok) {
                        if (!reader->ReadNext(want, &blk, err)) return DeviceMatF32View{};
                        cols = (blk.dtype == io::ColBlockDType::kU8) ? blk.X_u8.cols : blk.X_f32.cols;
                        if (cols <= 0) break;
                        col0 = blk.col0;
                        Xd = cuda->KmeansUploadAndNormalizeHostPtrBAsync(
                            slot,
                            blk.X_f32.data.data(),
                            /*ldB=*/blk.X_f32.rows,
                            /*rowsB=*/d,
                            /*colsB=*/cols,
                            /*stage_to_pinned=*/stage_blocks_to_pinned,
                            /*stage_s=*/nullptr,
                            /*h2d_s=*/nullptr,
                            /*kernel_s=*/nullptr);
                    }
                    else {
                        used_direct = true;
                        if (cols <= 0) break;
                        Xd = cuda->KmeansUploadAndNormalizePinnedAsync(
                            slot, pinned, /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols,
                            /*h2d_s=*/nullptr, /*kernel_s=*/nullptr);
                    }
                }

                if (cols <= 0 || !Xd.ptr) break;

                // Optional RVQ residual reconstruction for layer>0 (Prompt-06): stage codes on copy stream,
                // then project on compute stream after the slot-ready wait.
                if (g_rvq_hook && g_rvq_hook->upto_layer > 0) {
                    if (!g_rvq_hook->codes || !g_rvq_hook->codebooks) {
                        throw std::runtime_error(
                            "BuildKmeansllSampleOnDeviceFromReader(rvq): missing codes/codebooks.");
                    }
                    std::vector<const void*> codes_ptr(static_cast<std::size_t>(g_rvq_hook->upto_layer), nullptr);
                    std::vector<int> code_bytes(static_cast<std::size_t>(g_rvq_hook->upto_layer), 0);
                    for (int l = 0; l < g_rvq_hook->upto_layer; ++l) {
                        std::string local_err;
                        const RvqCodeSpan span = g_rvq_hook->codes->SpanBlock(l, col0, cols, &local_err);
                        if (!local_err.empty()) {
                            throw std::runtime_error(
                                "BuildKmeansllSampleOnDeviceFromReader(rvq): SpanBlock failed: " + local_err);
                        }
                        codes_ptr[static_cast<std::size_t>(l)] = span.ptr;
                        code_bytes[static_cast<std::size_t>(l)] = RvqCodeBytes(span.dtype);
                    }
                    cuda->RvqStageCodesBlockInCopyStreamAfterUpload(slot, cols, g_rvq_hook->upto_layer,
                                                                    codes_ptr.data(), code_bytes.data());
                }

                cuda->KmeansComputeWaitForUpload(slot);
                if (g_rvq_hook && g_rvq_hook->upto_layer > 0) {
                    cuda->RvqProjectXnormBlockOnComputeStream(slot, Xd, cols, g_rvq_hook->upto_layer);
                }
                cuda->KmeansllCopyActiveSlotToSample(slot, filled, cols);
                cuda->KmeansMarkUploadSlotConsumed(slot);

                filled += cols;
                (void)used_direct;
            }

            // Reset for subsequent full scans.
            if (!reader->Reset(err)) return DeviceMatF32View{};
            if (filled != sample_n) {
                if (err) *err = "BuildKmeansllSampleOnDeviceFromReader: failed to fill init sample (unexpected EOF).";
                return DeviceMatF32View{};
            }
            cuda->SyncCompute();
            return cuda->KmeansllSampleView();
        }

        void KmeansllUpdateBestDotWithNewIndices(CudaStreamKernels* cuda,
                                                 const DeviceMatF32View& dX,
                                                 const std::vector<int>& new_idx,
                                                 int tile_cols,
                                                 int center_batch) {
            if (!cuda || !dX.ptr || new_idx.empty()) return;
            const int d = dX.rows;
            const int n = dX.cols;
            tile_cols = std::max(1, tile_cols);
            tile_cols = std::min(tile_cols, n);
            center_batch = std::max(1, center_batch);

            std::vector<int> idx_batch;
            idx_batch.reserve(static_cast<std::size_t>(center_batch));
            for (std::size_t b0 = 0; b0 < new_idx.size(); b0 += static_cast<std::size_t>(center_batch)) {
                const std::size_t b1 = std::min(new_idx.size(), b0 + static_cast<std::size_t>(center_batch));
                idx_batch.assign(new_idx.begin() + static_cast<std::ptrdiff_t>(b0),
                                 new_idx.begin() + static_cast<std::ptrdiff_t>(b1));

                const DeviceVecI32View d_idx = cuda->KmeansllUploadIndicesHost(
                    idx_batch.data(), static_cast<int>(idx_batch.size()));
                const DeviceMatF32View dC = cuda->KmeansllGatherColumns(dX, d_idx, static_cast<int>(idx_batch.size()));

                ColMajorMatrix<float> C;
                cuda->DownloadF32(dC, &C);

                for (int col0 = 0; col0 < n; col0 += tile_cols) {
                    const int cols = std::min(tile_cols, n - col0);
                    const float* dB = dX.ptr + static_cast<std::size_t>(col0) * static_cast<std::size_t>(dX.ld);
                    DeviceMatF32View scores;
                    cuda->GemmDeviceDevicePtrB(/*transA=*/true, /*transB=*/false,
                                                          /*alpha=*/1.0f,
                                                          C,
                                                          dB,
                                                          /*ldB=*/dX.ld,
                                                          /*rowsB=*/d,
                                                          /*colsB=*/cols,
                                                          /*beta=*/0.0f,
                                                          &scores);
                    cuda->KmeansEnsureBlockArgmax(cols);
                    DeviceVecI32View d_assign = cuda->KmeansBlockAssignView();
                    DeviceVecF32View d_best = cuda->KmeansBlockBestView();
                    cuda->ArgmaxColsF32Device(scores, &d_assign, &d_best);
                    cuda->KmeansllUpdateBestDotMaxOffset(d_best, col0);
                }
            }
            cuda->SyncCompute();
        }

        KmeansllSeedOut InitializeCentersKmeansLLGpuFromReader(io::IColBlockReader* reader,
                                                               int k,
                                                               const KmeansConfig& cfg,
                                                               std::mt19937* rng,
                                                               CudaStreamKernels* cuda,
                                                               int block_cols,
                                                               bool stage_blocks_to_pinned,
                                                               bool profile_timing,
                                                               KmeansTiming* timing,
                                                               std::string* err) {
            KmeansllSeedOut out;
            if (!reader || !cuda || k <= 0) return out;
            const int d = reader->d();
            const std::int64_t n64 = reader->n();
            if (d <= 0 || n64 <= 0) return out;

            const int sample_n = static_cast<int>(
                std::min<std::int64_t>(std::max<std::int64_t>(1, cfg.init_samples), n64));
            const int rounds = std::min(16, std::max(1, cfg.kmeansll_rounds));
            const float oversample_factor = std::min(16.0f, std::max(1.0f, cfg.kmeansll_oversample));
            // Scalable kmeans++ (kmeans||): use l = oversample_factor * K so that each round samples O(K) candidates.
            const float oversample_l = oversample_factor * static_cast<float>(k);
            const int cap_default = std::min(std::max(k, 16 * k), 1'000'000);
            int cap = (cfg.kmeansll_candidate_cap > 0) ? cfg.kmeansll_candidate_cap : cap_default;
            cap = std::max(k, std::max(1, cap));
            cap = std::min(cap, sample_n);

            std::uint64_t base_seed = cfg.kmeansll_seed;
            if (base_seed == 0 && rng) {
                base_seed = (static_cast<std::uint64_t>((*rng)()) << 32) ^ static_cast<std::uint64_t>((*rng)());
            }

            LogInfo("[kmeans_seed] method=kmeans|| gpu=1 sample_n=" + std::to_string(sample_n) +
                " K=" + std::to_string(k) +
                " rounds=" + std::to_string(rounds) +
                " l_fac=" + FormatFloat(oversample_factor, 2) +
                " l_abs=" + std::to_string(static_cast<int>(oversample_l)) +
                " cap=" + std::to_string(cap));

            Timer t_sample;
            const DeviceMatF32View dX = BuildKmeansllSampleOnDeviceFromReader(reader, sample_n, block_cols, rng,
                                                                              cuda, stage_blocks_to_pinned, err);
            if (!dX.ptr || dX.cols != sample_n) {
                if (err && err->empty())
                    *err =
                        "InitializeCentersKmeansLLGpuFromReader: failed to build device sample.";
                return out;
            }
            if (profile_timing && timing) timing->init_sample_s += t_sample.ElapsedSeconds();

            // Best-dot init.
            cuda->KmeansllEnsureBestDot(sample_n);
            cuda->KmeansllSetBestDot(-INFINITY);

            // Candidate bookkeeping (host): use a byte mask for fast dedup.
            std::vector<std::uint8_t> seen(static_cast<std::size_t>(sample_n), 0);
            std::vector<int> cand_idx;
            cand_idx.reserve(static_cast<std::size_t>(cap));
            std::vector<int> new_idx;
            new_idx.reserve(4096);
            int picked_raw_total = 0;
            int picked_new_total = 0;
            int rounds_ran = 0;

            // Initial seed: 1 random point.
            {
                std::uniform_int_distribution<int> uni(0, std::max(0, sample_n - 1));
                const int idx0 = uni(*rng);
                cand_idx.push_back(idx0);
                seen[static_cast<std::size_t>(idx0)] = 1;
                new_idx.push_back(idx0);
                picked_raw_total += 1;
                picked_new_total += 1;
            }

            Timer t_pick;
            // Update best-dot with initial seed.
            KmeansllUpdateBestDotWithNewIndices(cuda, dX, new_idx, /*tile_cols=*/8192, /*center_batch=*/256);

            for (int r = 0; r < rounds; ++r) {
                if (static_cast<int>(cand_idx.size()) >= cap) break;
                const float phi = cuda->KmeansllComputePhiFromBestDot();
                if (!(phi > 0.0f)) break;
                rounds_ran = r + 1;

                const int remaining = cap - static_cast<int>(cand_idx.size());
                int picked_n = 0;
                const std::uint64_t round_seed = base_seed + 0x9e3779b97f4a7c15ULL * static_cast<std::uint64_t>(r);
                cuda->KmeansllSampleCandidates(oversample_l, phi, round_seed, remaining, &picked_n);

                std::vector<int> picked;
                cuda->KmeansllDownloadLastSampled(&picked);
                picked_raw_total += static_cast<int>(picked.size());
                if (picked.empty()) break;

                new_idx.clear();
                new_idx.reserve(picked.size());
                for (int idx : picked) {
                    if (idx < 0 || idx >= sample_n) {
                        LogWarn("KmeansllInit: GPU sampled index " + std::to_string(idx) +
                            " out of range [0, " + std::to_string(sample_n) +
                            "); possible device-side overflow or uninitialized memory.");
                        continue;
                    }
                    std::uint8_t& flag = seen[static_cast<std::size_t>(idx)];
                    if (flag) continue;
                    flag = 1;
                    cand_idx.push_back(idx);
                    new_idx.push_back(idx);
                    if (static_cast<int>(cand_idx.size()) >= cap) break;
                }
                picked_new_total += static_cast<int>(new_idx.size());
                if (new_idx.empty()) break;
                KmeansllUpdateBestDotWithNewIndices(cuda, dX, new_idx, /*tile_cols=*/8192, /*center_batch=*/256);
            }

            // Finalize to exactly K seeds: gather candidates to host and reuse CPU kmeans++ on candidates.
            const int cand_before_pad = static_cast<int>(cand_idx.size());
            if (static_cast<int>(cand_idx.size()) < k) {
                std::uniform_int_distribution<int> uni(0, std::max(0, sample_n - 1));
                while (static_cast<int>(cand_idx.size()) < k) cand_idx.push_back(uni(*rng));
            }
            const int padded = std::max(0, k - cand_before_pad);
            // Candidate pool size passed into the final kmeans++ "reduce" step.
            // `cap` is already a user-controlled upper bound (via train.kmeansll_candidate_cap), so
            // do not further clamp it to 16*K here; that would make `kmeansll_rounds` ineffective
            // once cand_unique exceeds 16*K (especially for small-K layers).
            const int M_cap = cap;
            const int cand_before_cap = static_cast<int>(cand_idx.size());
            if (static_cast<int>(cand_idx.size()) > M_cap) {
                cand_idx.resize(static_cast<std::size_t>(M_cap));
            }
            const int capped = std::max(0, cand_before_cap - static_cast<int>(cand_idx.size()));

            LogInfo("[kmeans_seed_pp] K=" + std::to_string(k) +
                " rounds_ran=" + std::to_string(rounds_ran) +
                " picked_raw=" + std::to_string(picked_raw_total) +
                " picked_new=" + std::to_string(picked_new_total) +
                " cand_unique=" + std::to_string(cand_before_pad) +
                " padded=" + std::to_string(padded) +
                " capped=" + std::to_string(capped) +
                " pp_input=" + std::to_string(static_cast<int>(cand_idx.size())));
            const DeviceVecI32View d_cand_idx = cuda->KmeansllUploadIndicesHost(
                cand_idx.data(), static_cast<int>(cand_idx.size()));
            const DeviceMatF32View dCand = cuda->KmeansllGatherColumns(dX, d_cand_idx,
                                                                       static_cast<int>(cand_idx.size()));
            ColMajorMatrix<float> Cand;
            cuda->DownloadF32(dCand, &Cand);
            out.centers = InitializeCentersKmeansPP(Cand, k, rng);
            NormalizeColumnsInPlace(&out.centers);

            // Precompute a per-cluster fallback sample on host to avoid keeping the full sample on host.
            out.fallback_sample = ColMajorMatrix<float>(d, k);
            {
                std::vector<int> fb_idx(static_cast<std::size_t>(k));
                std::uniform_int_distribution<int> uni(0, std::max(0, sample_n - 1));
                for (int c = 0; c < k; ++c) fb_idx[static_cast<std::size_t>(c)] = uni(*rng);
                const DeviceVecI32View d_fb = cuda->KmeansllUploadIndicesHost(fb_idx.data(), k);
                const DeviceMatF32View dFb = cuda->KmeansllGatherColumns(dX, d_fb, k);
                cuda->DownloadF32(dFb, &out.fallback_sample);
            }

            cuda->KmeansllReleaseSampleCache();
            if (profile_timing && timing) timing->init_pick_s += t_pick.ElapsedSeconds();
            return out;
        }
    } // namespace

    ColMajorMatrix<float> SphericalKmeansCentersOnlyStreamingReader(io::IColBlockReader* reader,
                                                                    int k,
                                                                    const KmeansConfig& cfg,
                                                                    std::mt19937* rng,
                                                                    StreamKernelProvider* stream_kernels,
                                                                    bool profile_timing,
                                                                    KmeansTiming* timing,
                                                                    std::string* err) {
        if (!reader) {
            if (err) *err = "SphericalKmeansCentersOnlyStreamingReader: reader is null.";
            return {};
        }
        const int d = reader->d();
        const std::int64_t n64 = reader->n();
        if (d <= 0 || n64 <= 0 || k <= 0) {
            return {d, std::max(0, k)};
        }
        const int block_cols = (cfg.block_cols > 0) ? cfg.block_cols : 200000;
        const bool anneal_enabled = AnnealingEnabled(cfg);
        PopulateKmeansTimingFlags(timing, cfg, k);

        Timer t_total;

        CudaStreamKernels* cuda_kernels = nullptr;
        if (stream_kernels && stream_kernels->IsGpu()) {
            cuda_kernels = dynamic_cast<CudaStreamKernels*>(stream_kernels);
        }
        if (cuda_kernels && g_rvq_hook && g_rvq_hook->upto_layer > 0) {
            if (!g_rvq_hook->codebooks) {
                if (err) *err = "SphericalKmeansCentersOnlyStreamingReader(rvq): codebooks is null.";
                return {};
            }
            cuda_kernels->RvqSetPriorCodebooks(*g_rvq_hook->codebooks, g_rvq_hook->upto_layer);
        }

        // NOTE: In this reader-based streaming path we interpret `cfg.pin_host_x` as:
        // "stage each host block into an internal pinned buffer before H2D".
        // If false, we upload directly from pageable host memory (slower H2D, but avoids the extra memcpy).
        const bool stage_blocks_to_pinned = (cuda_kernels != nullptr) && cfg.pin_host_x;
        if (timing && stage_blocks_to_pinned) {
            timing->used_pinned_host_x = 1;
        }

        ColMajorMatrix<float> sample_norm;
        ColMajorMatrix<float> centers;
        {
            Timer t_init;
            if (IsKmeansllInitMethod(cfg.init_method) && cuda_kernels && cfg.kmeansll_gpu_enable) {
                // kmeans|| seeding: build init_samples on GPU and run scalable kmeans++ rounds on device.
                KmeansllSeedOut seed = InitializeCentersKmeansLLGpuFromReader(reader,
                                                                              k,
                                                                              cfg,
                                                                              rng,
                                                                              cuda_kernels,
                                                                              block_cols,
                                                                              stage_blocks_to_pinned,
                                                                              profile_timing,
                                                                              timing,
                                                                              err);
                if (seed.centers.rows == d && seed.centers.cols == k &&
                    seed.fallback_sample.rows == d && seed.fallback_sample.cols == k) {
                    centers = std::move(seed.centers);
                    sample_norm = std::move(seed.fallback_sample);
                }
                else {
                    LogWarn("kmeans|| seeding failed; falling back to CPU init (random/kmeans++).");
                    Timer t_sample;
                    sample_norm = BuildNormalizedSampleFromReader(reader, cfg.init_samples, block_cols, rng, err);
                    if (profile_timing && timing) timing->init_sample_s += t_sample.ElapsedSeconds();
                    if (sample_norm.cols == 0) {
                        return {};
                    }
                    Timer t_pick;
                    centers = InitializeCentersFast(sample_norm, k, cfg, rng);
                    if (profile_timing && timing) timing->init_pick_s += t_pick.ElapsedSeconds();
                }
            }
            else {
                {
                    Timer t_sample;
                    // Optimization: For random init, only need k samples. For kmeans++, use full init_samples.
                    int effective_sample_n = cfg.init_samples;
                    if (cfg.init_method == "random") {
                        // Random only needs k samples; use 2*k for some redundancy.
                        effective_sample_n = std::min(cfg.init_samples, std::max(k, k * 2));
                    }
                    sample_norm = BuildNormalizedSampleFromReader(reader, effective_sample_n, block_cols, rng, err);
                    if (profile_timing && timing) timing->init_sample_s += t_sample.ElapsedSeconds();
                }
                if (sample_norm.cols == 0) {
                    return {};
                }
                {
                    Timer t_pick;
                    centers = InitializeCentersFast(sample_norm, k, cfg, rng);
                    if (profile_timing && timing) timing->init_pick_s += t_pick.ElapsedSeconds();
                }
            }
            if (profile_timing && timing) {
                timing->init_centers_s += t_init.ElapsedSeconds();
            }
        }

        // Fallback indices (into sample_norm) for empty clusters.
        std::vector<int> fallback(static_cast<std::size_t>(k), 0);
        if (IsKmeansllInitMethod(cfg.init_method) && cuda_kernels && cfg.kmeansll_gpu_enable &&
            sample_norm.rows == d && sample_norm.cols == k) {
            // In kmeans|| mode we precomputed a per-cluster fallback sample (d x k), one column per cluster.
            for (int c = 0; c < k; ++c) fallback[static_cast<std::size_t>(c)] = c;
        }
        else {
            std::uniform_int_distribution<int> uni(0, std::max(0, sample_norm.cols - 1));
            for (int c = 0; c < k; ++c) fallback[static_cast<std::size_t>(c)] = uni(*rng);
        }

        if (cuda_kernels) {
            cuda_kernels->KmeansSetCollectMetrics(cfg.collect_metrics);
            cuda_kernels->KmeansTimingEnable(profile_timing && (timing != nullptr));
            if (profile_timing && timing) {
                cuda_kernels->KmeansTimingReset();
            }
        }

        const auto ReaderReadNextTimed = [&](int max_cols, io::ColBlock* out) -> bool
        {
            Timer t_read;
            const bool ok = reader->ReadNext(max_cols, out, err);
            if (profile_timing && timing) {
                timing->reader_io_s += t_read.ElapsedSeconds();
                if (auto* rtim = dynamic_cast<io::IColBlockReaderTiming*>(reader)) {
                    const auto b = rtim->LastTimingBreakdown();
                    timing->reader_file_s += b.file_s;
                    timing->reader_unpack_s += b.unpack_s;
                }
            }
            return ok;
        };

        const auto TimingComputeBegin = [&](int slot)
        {
            if (profile_timing && timing && cuda_kernels) {
                cuda_kernels->KmeansTimingComputeSectionBegin(slot);
            }
        };
        const auto TimingComputeEnd = [&](int slot)
        {
            if (profile_timing && timing && cuda_kernels) {
                cuda_kernels->KmeansTimingComputeSectionEnd(slot);
            }
        };

        RvqInitCodesInMemory* emit_codes = (g_rvq_hook != nullptr) ? g_rvq_hook->emit_codes : nullptr;
        const int emit_layer = (g_rvq_hook != nullptr) ? g_rvq_hook->emit_layer : -1;
        const bool emit_codes_base_enabled = (emit_codes != nullptr && emit_layer >= 0);
        if (emit_codes_base_enabled) {
            if (emit_layer >= emit_codes->m()) {
                if (err) *err = "SphericalKmeansCentersOnlyStreamingReader: emit_layer out of range.";
                return {};
            }
            if (emit_codes->n() != n64) {
                if (err) *err = "SphericalKmeansCentersOnlyStreamingReader: emit_codes.n != reader.n.";
                return {};
            }
        }
        // Prompt-10: emit compact codes only once per layer (on the final update scan that is actually executed).
        // To avoid a post-training extra scan, we "arm" a single extra iteration when convergence is detected,
        // so the next iteration can emit codes and then stop.
        bool emit_codes_armed = false;
        bool emit_codes_done = false;

        // Device-resident anneal weights (best-effort). Used by:
        // - two-pass adaptive iterations (Prompt-01), and
        // - one-pass adaptive iterations (Prompt-02).
        const bool request_onepass = (cfg.anneal_mode == KmeansAnnealMode::kOnePass);
        const bool want_device_weights =
            (cuda_kernels != nullptr) && cfg.anneal_no_spill && cfg.anneal_weights_device && !cfg.
            force_disable_device_weights;
        const bool use_device_weights =
            (anneal_enabled && want_device_weights)
                ? cuda_kernels->KmeansInitAnnealWeightsAll(n64, cfg.initial_weight)
                : false;
        const bool onepass_enabled = request_onepass && use_device_weights;
        static bool warned_onepass_fallback = false;
        if (request_onepass && !onepass_enabled && anneal_enabled && !warned_onepass_fallback) {
            warned_onepass_fallback = true;
            LogWarn("kmeans anneal_mode=onepass requested, but device weights unavailable; falling back to twopass.");
        }

        float threshold_prev = cfg.cost_threshold;

        std::filesystem::path w_in_path;
        std::filesystem::path w_out_path;
        if (anneal_enabled && !use_device_weights) {
            const std::filesystem::path tmp_dir =
                !cfg.tmp_dir.empty() ? std::filesystem::path(cfg.tmp_dir) : std::filesystem::path("./tmp");
            std::error_code ec;
            std::filesystem::create_directories(tmp_dir, ec);
            if (ec) {
                if (err) *err = "Failed to create kmeans tmp_dir: " + ec.message();
                return {};
            }
            w_in_path = tmp_dir / "kmeans_weights_in_f32.bin";
            w_out_path = tmp_dir / "kmeans_weights_out_f32.bin";
            std::filesystem::remove(w_out_path, ec);
            WriteConstantF32VectorOrThrow(w_in_path.string(), static_cast<std::size_t>(n64), cfg.initial_weight);
        }

        ColMajorMatrix<float> sum_x(d, k);
        std::vector<float> sum_w(static_cast<std::size_t>(k), 0.0f);
        std::vector<int> assign_blk;
        std::vector<float> dot_blk;
        io::ColBlock blk;

        struct UploadedBlock
        {
            std::int64_t col0 = 0;
            int cols = 0;
            io::ColBlockDType dtype = io::ColBlockDType::kF32;
            DeviceMatF32View Xd;
        };

        bool direct_into_known = false;
        bool direct_into_supported = false;

        // Optional partial device cache (prefix window) for streaming k-means.
        const bool want_partial_dev_cache =
            (cuda_kernels != nullptr) &&
            !cfg.cache_xnorm_device &&
            (cfg.device_cache_mb > 0) &&
            (cfg.max_iters > 1); // best-effort: only useful when we scan multiple iterations
        std::int64_t partial_cache_cols = 0;
        std::int64_t partial_cache_cursor = 0;
        bool partial_cache_ready = false;
        auto BeginScanWithPartialCache = [&]() -> bool
        {
            partial_cache_cursor = 0;
            if (!reader->Reset(err)) return false;
            if (partial_cache_cols > 0 && partial_cache_cols < n64) {
                std::string seek_err;
                if (!reader->Seek(partial_cache_cols, &seek_err)) {
                    LogWarn("kmeans partial device cache disabled: reader->Seek failed: " + seek_err);
                    if (cuda_kernels) cuda_kernels->KmeansReleasePartialXnormCache();
                    partial_cache_ready = false;
                    partial_cache_cols = 0;
                    return true; // continue without cache
                }
            }
            return true;
        };

        auto ReadAndUploadNextPinned = [&](int slot, UploadedBlock* out) -> bool
        {
            if (!out) {
                if (err) *err = "ReadAndUploadNextPinned: out is null.";
                return false;
            }
            if (!cuda_kernels) {
                if (err) *err = "ReadAndUploadNextPinned: cuda_kernels is null.";
                return false;
            }
            out->col0 = 0;
            out->cols = 0;
            out->dtype = reader->dtype();
            out->Xd = DeviceMatF32View{};

            const auto UploadFromPinned = [&](const void* pinned_ptr, int cols,
                                              io::ColBlockDType dtype) -> DeviceMatF32View
            {
                if (cols <= 0) return DeviceMatF32View{};
                if (dtype == io::ColBlockDType::kU8) {
                    return cuda_kernels->KmeansUploadU8AndNormalizePinnedAsync(
                        slot,
                        static_cast<const std::uint8_t*>(pinned_ptr),
                        /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols,
                        /*h2d_s=*/nullptr,
                        /*kernel_s=*/nullptr);
                }
                return cuda_kernels->KmeansUploadAndNormalizePinnedAsync(
                    slot,
                    static_cast<const float*>(pinned_ptr),
                    /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols,
                    /*h2d_s=*/nullptr,
                    /*kernel_s=*/nullptr);
            };

            const auto MaybeApplyRvqAfterUpload = [&](int cols)
            {
                if (!g_rvq_hook || g_rvq_hook->upto_layer <= 0 || cols <= 0) return;
                if (!g_rvq_hook->codes || !g_rvq_hook->codebooks) {
                    throw std::runtime_error("ReadAndUploadNextPinned(rvq): missing codes/codebooks.");
                }
                if (g_rvq_hook->upto_layer > static_cast<int>(g_rvq_hook->codebooks->size())) {
                    throw std::runtime_error("ReadAndUploadNextPinned(rvq): codebooks too small for upto_layer.");
                }
                std::vector<const void*> codes_ptr(static_cast<std::size_t>(g_rvq_hook->upto_layer), nullptr);
                std::vector<int> code_bytes(static_cast<std::size_t>(g_rvq_hook->upto_layer), 0);
                for (int l = 0; l < g_rvq_hook->upto_layer; ++l) {
                    std::string local_err;
                    const RvqCodeSpan span = g_rvq_hook->codes->SpanBlock(l, out->col0, cols, &local_err);
                    if (!local_err.empty()) {
                        throw std::runtime_error("ReadAndUploadNextPinned(rvq): SpanBlock failed: " + local_err);
                    }
                    codes_ptr[static_cast<std::size_t>(l)] = span.ptr;
                    code_bytes[static_cast<std::size_t>(l)] = RvqCodeBytes(span.dtype);
                }
                cuda_kernels->RvqStageCodesBlockInCopyStreamAfterUpload(
                    slot, cols, g_rvq_hook->upto_layer, codes_ptr.data(), code_bytes.data());
            };

            if (partial_cache_ready && partial_cache_cols > 0 && partial_cache_cursor < partial_cache_cols) {
                const int cols =
                    static_cast<int>(std::min<std::int64_t>(block_cols, partial_cache_cols - partial_cache_cursor));
                out->col0 = partial_cache_cursor;
                out->cols = cols;
                out->Xd = cuda_kernels->
                    KmeansCopyPartialCacheToSlot(slot, static_cast<int>(partial_cache_cursor), cols);
                MaybeApplyRvqAfterUpload(cols);
                partial_cache_cursor += cols;
                return true;
            }
            if (partial_cache_ready && partial_cache_cols > 0 && partial_cache_cols == n64 &&
                partial_cache_cursor >= partial_cache_cols) {
                // Cached the full scan range; do not re-read from the underlying reader.
                out->cols = 0;
                return true;
            }

            // Probe direct-into once; interpret "false + empty err" as unsupported.
            if (!direct_into_known) {
                std::string probe_err;
                if (out->dtype == io::ColBlockDType::kU8) {
                    std::uint8_t* pinned = cuda_kernels->KmeansGetPinnedStageBufferU8(slot, d, block_cols);
                    const std::size_t bytes =
                        sizeof(std::uint8_t) * static_cast<std::size_t>(d) * static_cast<std::size_t>(block_cols);
                    Timer t_read;
                    const bool ok = reader->ReadNextInto(block_cols, pinned, bytes, &out->col0, &out->cols, &probe_err);
                    if (profile_timing && timing) {
                        timing->reader_io_s += t_read.ElapsedSeconds();
                        if (auto* rtim = dynamic_cast<io::IColBlockReaderTiming*>(reader)) {
                            const auto b = rtim->LastTimingBreakdown();
                            timing->reader_file_s += b.file_s;
                            timing->reader_unpack_s += b.unpack_s;
                        }
                    }
                    if (ok) {
                        direct_into_known = true;
                        direct_into_supported = true;
                        if (out->cols > 0) {
                            out->Xd = UploadFromPinned(pinned, out->cols, out->dtype);
                            MaybeApplyRvqAfterUpload(out->cols);
                        }
                        else {
                            out->Xd = {};
                        }
                        return true;
                    }
                    if (!probe_err.empty()) {
                        if (err) *err = probe_err;
                        return false;
                    }
                }
                else {
                    float* pinned = cuda_kernels->KmeansGetPinnedStageBuffer(slot, d, block_cols);
                    const std::size_t bytes =
                        sizeof(float) * static_cast<std::size_t>(d) * static_cast<std::size_t>(block_cols);
                    Timer t_read;
                    const bool ok = reader->ReadNextInto(block_cols, pinned, bytes, &out->col0, &out->cols, &probe_err);
                    if (profile_timing && timing) {
                        timing->reader_io_s += t_read.ElapsedSeconds();
                        if (auto* rtim = dynamic_cast<io::IColBlockReaderTiming*>(reader)) {
                            const auto b = rtim->LastTimingBreakdown();
                            timing->reader_file_s += b.file_s;
                            timing->reader_unpack_s += b.unpack_s;
                        }
                    }
                    if (ok) {
                        direct_into_known = true;
                        direct_into_supported = true;
                        if (out->cols > 0) {
                            out->Xd = UploadFromPinned(pinned, out->cols, out->dtype);
                            MaybeApplyRvqAfterUpload(out->cols);
                        }
                        else {
                            out->Xd = {};
                        }
                        return true;
                    }
                    if (!probe_err.empty()) {
                        if (err) *err = probe_err;
                        return false;
                    }
                }
                direct_into_known = true;
                direct_into_supported = false;
            }

            if (direct_into_supported) {
                std::string local_err;
                if (out->dtype == io::ColBlockDType::kU8) {
                    std::uint8_t* pinned = cuda_kernels->KmeansGetPinnedStageBufferU8(slot, d, block_cols);
                    const std::size_t bytes =
                        sizeof(std::uint8_t) * static_cast<std::size_t>(d) * static_cast<std::size_t>(block_cols);
                    Timer t_read;
                    const bool ok = reader->ReadNextInto(block_cols, pinned, bytes, &out->col0, &out->cols, &local_err);
                    if (profile_timing && timing) {
                        timing->reader_io_s += t_read.ElapsedSeconds();
                        if (auto* rtim = dynamic_cast<io::IColBlockReaderTiming*>(reader)) {
                            const auto b = rtim->LastTimingBreakdown();
                            timing->reader_file_s += b.file_s;
                            timing->reader_unpack_s += b.unpack_s;
                        }
                    }
                    if (!ok) {
                        if (err) *err = !local_err.empty() ? local_err : "ReadNextInto failed.";
                        return false;
                    }
                    if (out->cols > 0) {
                        out->Xd = UploadFromPinned(pinned, out->cols, out->dtype);
                        MaybeApplyRvqAfterUpload(out->cols);
                    }
                    else {
                        out->Xd = {};
                    }
                    return true;
                }
                float* pinned = cuda_kernels->KmeansGetPinnedStageBuffer(slot, d, block_cols);
                const std::size_t bytes =
                    sizeof(float) * static_cast<std::size_t>(d) * static_cast<std::size_t>(block_cols);
                Timer t_read;
                const bool ok = reader->ReadNextInto(block_cols, pinned, bytes, &out->col0, &out->cols, &local_err);
                if (profile_timing && timing) {
                    timing->reader_io_s += t_read.ElapsedSeconds();
                    if (auto* rtim = dynamic_cast<io::IColBlockReaderTiming*>(reader)) {
                        const auto b = rtim->LastTimingBreakdown();
                        timing->reader_file_s += b.file_s;
                        timing->reader_unpack_s += b.unpack_s;
                    }
                }
                if (!ok) {
                    if (err) *err = !local_err.empty() ? local_err : "ReadNextInto failed.";
                    return false;
                }
                if (out->cols > 0) {
                    out->Xd = UploadFromPinned(pinned, out->cols, out->dtype);
                    MaybeApplyRvqAfterUpload(out->cols);
                }
                else {
                    out->Xd = {};
                }
                return true;
            }

            // Fallback: ReadNext + memcpy into pinned.
            io::ColBlock b;
            Timer t_read;
            if (!reader->ReadNext(block_cols, &b, err)) return false;
            if (profile_timing && timing) {
                timing->reader_io_s += t_read.ElapsedSeconds();
                if (auto* rtim = dynamic_cast<io::IColBlockReaderTiming*>(reader)) {
                    const auto bb = rtim->LastTimingBreakdown();
                    timing->reader_file_s += bb.file_s;
                    timing->reader_unpack_s += bb.unpack_s;
                }
            }
            out->dtype = b.dtype;
            out->col0 = b.col0;
            out->cols = (b.dtype == io::ColBlockDType::kU8) ? b.X_u8.cols : b.X_f32.cols;
            if (out->cols <= 0) return true;
            if (b.dtype == io::ColBlockDType::kU8) {
                std::uint8_t* pinned = cuda_kernels->KmeansGetPinnedStageBufferU8(slot, d, out->cols);
                std::memcpy(pinned,
                            b.X_u8.data.data(),
                            sizeof(std::uint8_t) * static_cast<std::size_t>(d) * static_cast<std::size_t>(out->cols));
                out->Xd = UploadFromPinned(pinned, out->cols, out->dtype);
                MaybeApplyRvqAfterUpload(out->cols);
                return true;
            }
            float* pinned = cuda_kernels->KmeansGetPinnedStageBuffer(slot, d, out->cols);
            std::memcpy(pinned,
                        b.X_f32.data.data(),
                        sizeof(float) * static_cast<std::size_t>(d) * static_cast<std::size_t>(out->cols));
            out->Xd = UploadFromPinned(pinned, out->cols, out->dtype);
            MaybeApplyRvqAfterUpload(out->cols);
            return true;
        };

        if (want_partial_dev_cache) {
            const std::int64_t bytes_budget =
                static_cast<std::int64_t>(cfg.device_cache_mb) * 1024ll * 1024ll;
            const std::int64_t bytes_per_col = static_cast<std::int64_t>(d) * static_cast<std::int64_t>(sizeof(float));
            const std::int64_t cols_budget = (bytes_per_col > 0) ? (bytes_budget / bytes_per_col) : 0;
            partial_cache_cols = std::min<std::int64_t>(n64, std::max<std::int64_t>(0, cols_budget));
            if (partial_cache_cols > 0) {
                try {
                    cuda_kernels->KmeansEnsurePartialXnormCache(d, static_cast<int>(partial_cache_cols));
                }
                catch (...) {
                    partial_cache_cols = 0;
                }
            }
            if (partial_cache_cols > 0) {
                // Fill the device cache once: read+upload+normalize on the copy stream, then D2D copy into cache.
                if (!reader->Reset(err)) return {};
                std::int64_t filled = 0;
                while (filled < partial_cache_cols) {
                    const int want = static_cast<int>(std::min<std::int64_t>(block_cols, partial_cache_cols - filled));
                    UploadedBlock cur;
                    if (!ReadAndUploadNextPinned(/*slot=*/0, &cur)) return {};
                    if (cur.cols <= 0) break;
                    const int copy_cols = std::min(cur.cols, want);
                    cuda_kernels->KmeansCopyActiveSlotToPartialCache(/*slot=*/0, static_cast<int>(filled), copy_cols);
                    filled += copy_cols;
                }
                cuda_kernels->Sync();
                if (filled != partial_cache_cols) {
                    cuda_kernels->KmeansReleasePartialXnormCache();
                    partial_cache_cols = 0;
                }
                else {
                    // After cache fill, reset for the first scan; BeginScanWithPartialCache will Seek for remainder.
                    if (!reader->Reset(err)) return {};
                    partial_cache_ready = true;
                    if (!g_rvq_hook || g_rvq_hook->upto_layer == 0) {
                        const std::int64_t cache_bytes =
                            partial_cache_cols * static_cast<std::int64_t>(d) * static_cast<std::int64_t>(sizeof(
                                float));
                        const std::int64_t cache_mb = cache_bytes / (1024ll * 1024ll);
                        LogInfo("[kmeans_cache] device_cache_mb=" + std::to_string(cfg.device_cache_mb) +
                            " cache_mb=" + std::to_string(cache_mb) +
                            " cache_cols=" + std::to_string(partial_cache_cols) +
                            " d=" + std::to_string(d) +
                            " block_cols=" + std::to_string(block_cols) +
                            " n=" + std::to_string(n64));
                    }
                }
            }
        }

        double prev_cost = std::numeric_limits<double>::infinity();
        for (int iter = 1; iter <= cfg.max_iters; ++iter) {
            const bool adaptive_active = (anneal_enabled && iter > cfg.warmup_iters);
            const bool onepass_active = adaptive_active && onepass_enabled;
            const bool twopass_active = adaptive_active && !onepass_active;
            int scans_this_iter = 0;
            const bool use_hier2 = ShouldUseHier2(k, cfg) && cfg.large_k_hier2_train;
            Hier2Split hier2;
            ColMajorMatrix<float> coarse;
            if (use_hier2) {
                hier2 = ComputeHier2Split(k, cfg);
                coarse = BuildCoarseFromFineCPU(centers, k, hier2.K1, hier2.K2);
                if (timing) {
                    timing->used_hier2 = 1;
                    timing->hier2_K = k;
                    timing->hier2_K1 = hier2.K1;
                    timing->hier2_K2 = hier2.K2;
                }
            }

            if (cuda_kernels) {
                cuda_kernels->KmeansResetSums(d, k);
                if (cfg.collect_metrics) {
                    cuda_kernels->KmeansResetMetricTotals();
                }
                if (use_device_weights || (adaptive_active && anneal_enabled)) {
                    cuda_kernels->KmeansResetAnnealTotals();
                }
            }
            sum_x.data.assign(static_cast<std::size_t>(d) * static_cast<std::size_t>(k), 0.0f);
            std::fill(sum_w.begin(), sum_w.end(), 0.0f);

            // Prompt-10: emit codes only once, on the last update scan that is actually executed:
            // - if we hit max_iters, emit on that last iter
            // - if we early-stop, do one extra iter (armed) to emit and then stop
            const bool emit_codes_this_iter =
                emit_codes_base_enabled && !emit_codes_done && (emit_codes_armed || iter == cfg.max_iters);
            if (emit_codes_this_iter) {
                std::string emit_err;
                if (!emit_codes->ResetWriteCursor(emit_layer, &emit_err)) {
                    if (err) *err = "SphericalKmeansCentersOnlyStreamingReader: ResetWriteCursor failed: " + emit_err;
                    return {};
                }
            }

            double total_cost = 0.0;
            double total_w = 0.0;

            const int hist_bins = adaptive_active ? std::max(1, cfg.quantile_bins) : 0;

            float effective_threshold = cfg.cost_threshold;
            float threshold_next = threshold_prev;
            float annealed_factor = 0.0f;
            if (adaptive_active) {
                if (cuda_kernels) cuda_kernels->KmeansResetCostHistogram(hist_bins);
                std::vector<std::uint64_t> hist_u64;
                if (!cuda_kernels) {
                    hist_u64.assign(static_cast<std::size_t>(hist_bins), 0);
                }
                const float lo = 0.0f;
                const float hi = 2.0f;
                const float inv_span = static_cast<float>(hist_bins) / (hi - lo);

                const float denom = static_cast<float>(std::max(1, cfg.max_iters - cfg.warmup_iters));
                annealed_factor = cfg.annealing_factor * (1.0f - std::exp(-(iter - cfg.warmup_iters) / denom));

                if (twopass_active) {
                    if (partial_cache_ready) {
                        if (!BeginScanWithPartialCache()) return {};
                    }
                    else {
                        if (!reader->Reset(err)) return {};
                    }
                    ++scans_this_iter;
                    // Pass 1 (two-pass): build histogram of per-sample costs (device best-dot) to estimate quantile threshold.
                    //
                    // Performance: when CUDA + pinned staging is enabled, overlap H2D+normalize (copy stream)
                    // with assignment/histogram accumulation (compute stream) via 2-slot ping-pong.
                    const bool overlap_h2d_pass1 =
                        (cuda_kernels != nullptr) && stage_blocks_to_pinned;
                    if (timing && overlap_h2d_pass1) {
                        timing->used_overlap_pass1 = 1;
                    }
                    if (overlap_h2d_pass1) {
                        UploadedBlock cur;
                        UploadedBlock next;
                        DeviceMatF32View Xd_slot[2];
                        int slot_cur = 0;
                        int slot_next = 1;
                        if (!ReadAndUploadNextPinned(slot_cur, &cur)) return {};
                        if (cur.cols > 0) {
                            Xd_slot[slot_cur] = cur.Xd;
                            while (true) {
                                const int cols_cur = cur.cols;
                                if (cols_cur <= 0) break;
                                if (!ReadAndUploadNextPinned(slot_next, &next)) return {};
                                if (next.cols > 0) {
                                    Xd_slot[slot_next] = next.Xd;
                                }

                                cuda_kernels->KmeansComputeWaitForUpload(slot_cur);
                                const DeviceMatF32View Xd = Xd_slot[slot_cur];
                                if (g_rvq_hook && g_rvq_hook->upto_layer > 0) {
                                    cuda_kernels->RvqProjectXnormBlockOnComputeStream(
                                        slot_cur, Xd, cols_cur, g_rvq_hook->upto_layer);
                                }
                                TimingComputeBegin(slot_cur);
                                if (use_hier2) {
                                    cuda_kernels->KmeansHier2AssignColsF32Device(Xd, coarse, centers,
                                        /*K_valid=*/k,
                                        /*K1=*/hier2.K1,
                                        /*K2=*/hier2.K2,
                                        /*topL=*/cfg.large_k_top_coarse);
                                }
                                else {
                                    AssignBlockGpuTiled(Xd, centers, k, stream_kernels, cuda_kernels,
                                                        /*profile_timing=*/false, /*timing=*/nullptr,
                                                        /*need_host_argmax=*/false,
                                                        /*assign_blk=*/nullptr,
                                                        /*dot_blk=*/nullptr);
                                }
                                cuda_kernels->KmeansAccumulateCostHistogramFromBestDot(
                                    cuda_kernels->KmeansBlockBestView(), hist_bins);
                                TimingComputeEnd(slot_cur);
                                cuda_kernels->KmeansMarkUploadSlotConsumed(slot_cur);

                                cur = next;
                                std::swap(slot_cur, slot_next);
                            }
                        }
                    }
                    else {
                        while (true) {
                            if (!ReaderReadNextTimed(block_cols, &blk)) return {};
                            const int cols_blk =
                                (blk.dtype == io::ColBlockDType::kU8) ? blk.X_u8.cols : blk.X_f32.cols;
                            if (cols_blk == 0) break;

                            if (cuda_kernels) {
                                const DeviceMatF32View Xd = UploadAndNormalizeBlock(cuda_kernels, blk,
                                    stage_blocks_to_pinned,
                                    profile_timing, timing);
                                TimingComputeBegin(0);
                                if (use_hier2) {
                                    Timer t_assign;
                                    cuda_kernels->KmeansHier2AssignColsF32Device(Xd, coarse, centers,
                                        /*K_valid=*/k,
                                        /*K1=*/hier2.K1,
                                        /*K2=*/hier2.K2,
                                        /*topL=*/cfg.large_k_top_coarse);
                                    if (profile_timing && timing) {
                                        stream_kernels->SyncCompute();
                                        timing->assign_gemm_s += t_assign.ElapsedSeconds();
                                    }
                                }
                                else {
                                    AssignBlockGpuTiled(Xd, centers, k, stream_kernels, cuda_kernels,
                                                        profile_timing, timing,
                                                        /*need_host_argmax=*/false,
                                                        /*assign_blk=*/nullptr,
                                                        /*dot_blk=*/nullptr);
                                }
                                cuda_kernels->KmeansAccumulateCostHistogramFromBestDot(
                                    cuda_kernels->KmeansBlockBestView(), hist_bins);
                                TimingComputeEnd(0);
                                if (blk.dtype == io::ColBlockDType::kU8) {
                                    cuda_kernels->KmeansMarkUploadSlotConsumed(0);
                                }
                                // NOTE: no per-block SyncCompute here; it destroys H2D/compute overlap and can dominate wall time.
                            }
                            else {
                                ColMajorMatrix<float> tmp_f32;
                                ColMajorMatrix<float>* Xblk = nullptr;
                                if (blk.dtype == io::ColBlockDType::kU8) {
                                    tmp_f32 = ColMajorMatrix<float>(d, cols_blk);
                                    for (int j = 0; j < cols_blk; ++j) {
                                        const std::uint8_t* xj = blk.X_u8.Col(j);
                                        float* dst = tmp_f32.Col(j);
#pragma omp simd
                                        for (int r = 0; r < d; ++r) dst[r] = static_cast<float>(xj[r]);
                                    }
                                    Xblk = &tmp_f32;
                                }
                                else {
                                    Xblk = &blk.X_f32;
                                }
                                NormalizeColumnsInPlace(Xblk);
                                AssignSamplesGpu(*Xblk, centers, stream_kernels, profile_timing, timing,
                                                 &assign_blk, &dot_blk);
                                for (int j = 0; j < cols_blk; ++j) {
                                    float x = 1.0f - dot_blk[static_cast<std::size_t>(j)];
                                    if (x < lo) x = lo;
                                    if (x > hi) x = hi;
                                    int b = static_cast<int>((x - lo) * inv_span);
                                    if (b >= hist_bins) b = hist_bins - 1;
                                    hist_u64[static_cast<std::size_t>(b)] += 1;
                                }
                            }
                        }
                    }

                    const std::uint64_t target = static_cast<std::uint64_t>(
                        cfg.outlier_quantile * static_cast<float>(std::max<std::int64_t>(1, n64 - 1)));
                    std::uint64_t prefix = 0;
                    int bq = hist_bins - 1;
                    if (cuda_kernels) {
                        std::vector<std::uint32_t> hist_u32(static_cast<std::size_t>(hist_bins));
                        // Async D2H so we don't hard-sync the compute stream here.
                        cuda_kernels->KmeansDownloadCostHistogramAsync(hist_u32.data(), hist_bins);
                        cuda_kernels->KmeansDownloadCostHistogramSync();
                        for (int b = 0; b < hist_bins; ++b) {
                            prefix += static_cast<std::uint64_t>(hist_u32[static_cast<std::size_t>(b)]);
                            if (prefix > target) {
                                bq = b;
                                break;
                            }
                        }
                    }
                    else {
                        for (int b = 0; b < hist_bins; ++b) {
                            prefix += hist_u64[static_cast<std::size_t>(b)];
                            if (prefix > target) {
                                bq = b;
                                break;
                            }
                        }
                    }
                    const float bw = (hi - lo) / static_cast<float>(hist_bins);
                    const float quantile_threshold = lo + (static_cast<float>(bq) + 0.5f) * bw;
                    effective_threshold = std::max(quantile_threshold, cfg.cost_threshold);
                }
                else {
                    // One-pass: update weights using the previous iteration's threshold, and build the histogram
                    // for the next iteration during the (single) assignment scan.
                    effective_threshold = threshold_prev;
                }
            }

            std::ifstream f_w_in;
            std::ofstream f_w_out;
            const bool need_weight_files = adaptive_active && !use_device_weights;
            if (need_weight_files) {
                f_w_in.open(w_in_path, std::ios::binary);
                if (!f_w_in) {
                    if (err) *err = "Failed to open weights_in: " + w_in_path.string();
                    return {};
                }
                f_w_out.open(w_out_path, std::ios::binary | std::ios::trunc);
                if (!f_w_out) {
                    if (err) *err = "Failed to open weights_out: " + w_out_path.string();
                    return {};
                }
            }

            if (partial_cache_ready) {
                if (!BeginScanWithPartialCache()) return {};
            }
            else {
                if (!reader->Reset(err)) return {};
            }
            ++scans_this_iter;
            std::vector<int> offsets;
            std::vector<int> cursor;
            std::vector<int> order;
            const bool overlap_h2d_pass2 =
                (cuda_kernels != nullptr) && stage_blocks_to_pinned;
            if (timing && overlap_h2d_pass2) {
                timing->used_overlap_pass2 = 1;
            }
            struct EmitPending
            {
                bool pending = false;
                std::int64_t col0 = 0;
                int cols = 0;
            };
            EmitPending emit_pending[2];
            auto EmitFlushSlot = [&](int slot) -> bool
            {
                if (!emit_codes_this_iter || !cuda_kernels) return true;
                if (slot < 0 || slot >= 2) return true;
                if (!emit_pending[slot].pending) return true;
                const int* assign_ptr = nullptr;
                int cols_sync = 0;
                cuda_kernels->KmeansDownloadBlockAssignSync(slot, &assign_ptr, &cols_sync);
                if (!assign_ptr || cols_sync != emit_pending[slot].cols) {
                    if (err) *err = "SphericalKmeansCentersOnlyStreamingReader: async codes download failed.";
                    return false;
                }
                std::string emit_err;
                if (!emit_codes->WriteBlockFromI32(emit_layer, emit_pending[slot].col0, emit_pending[slot].cols,
                                                   assign_ptr, &emit_err)) {
                    if (err)
                        *err = "SphericalKmeansCentersOnlyStreamingReader: codes_store.WriteBlockFromI32 failed: "
                            + emit_err;
                    return false;
                }
                emit_pending[slot].pending = false;
                return true;
            };
            auto EmitScheduleSlot = [&](int slot, std::int64_t col0, int cols) -> bool
            {
                if (!emit_codes_this_iter || !cuda_kernels) return true;
                if (slot < 0 || slot >= 2) return true;
                // IMPORTANT for overlap: flush the previous pending slot only when we are about to reuse
                // this slot for the next async download. Doing it earlier can block H2D scheduling and
                // collapse GPU utilization.
                if (!EmitFlushSlot(slot)) return false;
                cuda_kernels->KmeansDownloadBlockAssignAsync(slot);
                emit_pending[slot].pending = true;
                emit_pending[slot].col0 = col0;
                emit_pending[slot].cols = cols;
                return true;
            };
            if (overlap_h2d_pass2) {
                UploadedBlock cur;
                UploadedBlock next;
                if (!ReadAndUploadNextPinned(/*slot=*/0, &cur)) return {};
                if (cur.cols > 0) {
                    std::vector<float> w_cur;
                    std::vector<float> w_next;
                    if (need_weight_files) {
                        ReadVectorExactOrThrow(f_w_in, &w_cur, static_cast<std::size_t>(cur.cols));
                    }

                    DeviceMatF32View Xd_slot[2];
                    int slot_cur = 0;
                    int slot_next = 1;
                    Xd_slot[slot_cur] = cur.Xd;
                    while (true) {
                        const int cols_cur = cur.cols;
                        if (cols_cur <= 0) break;
                        if (!ReadAndUploadNextPinned(slot_next, &next)) return {};
                        w_next.clear();
                        if (need_weight_files && next.cols > 0) {
                            ReadVectorExactOrThrow(f_w_in, &w_next, static_cast<std::size_t>(next.cols));
                        }
                        if (next.cols > 0) {
                            Xd_slot[slot_next] = next.Xd;
                        }

                        cuda_kernels->KmeansComputeWaitForUpload(slot_cur);
                        const DeviceMatF32View Xd = Xd_slot[slot_cur];
                        if (g_rvq_hook && g_rvq_hook->upto_layer > 0) {
                            cuda_kernels->RvqProjectXnormBlockOnComputeStream(
                                slot_cur, Xd, cols_cur, g_rvq_hook->upto_layer);
                        }
                        TimingComputeBegin(slot_cur);
                        if (use_hier2) {
                            cuda_kernels->KmeansHier2AssignColsF32Device(Xd, coarse, centers,
                                                                         /*K_valid=*/k,
                                                                         /*K1=*/hier2.K1,
                                                                         /*K2=*/hier2.K2,
                                                                         /*topL=*/cfg.large_k_top_coarse);
                        }
                        else {
                            AssignBlockGpuTiled(Xd, centers, k, stream_kernels, cuda_kernels,
                                                /*profile_timing=*/false, /*timing=*/nullptr,
                                                /*need_host_argmax=*/false,
                                                /*assign_blk=*/nullptr,
                                                /*dot_blk=*/nullptr);
                        }
                        if (!EmitScheduleSlot(slot_cur, cur.col0, cols_cur)) return {};

                        if (cfg.collect_metrics) {
                            cuda_kernels->KmeansAccumulateMseFromBlockBestDot(cols_cur);
                        }
                        if (onepass_active) {
                            cuda_kernels->KmeansAccumulateCostHistogramFromBestDot(
                                cuda_kernels->KmeansBlockBestView(), hist_bins);
                        }
                        if (use_device_weights) {
                            if (adaptive_active) {
                                cuda_kernels->KmeansAnnealUpdateWeightsAndTotalsFromBlockBest(
                                    cur.col0, cols_cur, effective_threshold, annealed_factor, cfg.min_weight);
                            }
                            else {
                                const DeviceVecF32View d_w =
                                    cuda_kernels->KmeansAnnealWeightsBlockView(cur.col0, cols_cur);
                                cuda_kernels->KmeansAccumulateAnnealTotalsFromBlockBestDotAndWeights(d_w);
                            }
                            const DeviceVecF32View d_w =
                                cuda_kernels->KmeansAnnealWeightsBlockView(cur.col0, cols_cur);
                            cuda_kernels->KmeansAccumulateWeightsDeviceAssignNoAtomic(
                                Xd, cuda_kernels->KmeansBlockAssignView(), d_w);
                        }
                        else if (adaptive_active) {
                            cuda_kernels->KmeansComputeOutlierMaskFromBlockBest(cols_cur, effective_threshold);
                            cuda_kernels->KmeansDownloadOutlierMaskAsync(cols_cur);
                            std::uint8_t* mask_ptr = nullptr;
                            int mask_bytes = 0;
                            cuda_kernels->KmeansSyncOutlierMaskDownload(&mask_ptr, &mask_bytes);
                            if (!mask_ptr || mask_bytes < cols_cur) {
                                if (err) *err = "Kmeans outlier mask download failed.";
                                return {};
                            }
                            for (int j = 0; j < cols_cur; ++j) {
                                float w = w_cur[static_cast<std::size_t>(j)];
                                if (mask_ptr[j]) {
                                    w = std::max(cfg.min_weight, w * (1.0f - annealed_factor));
                                }
                                else {
                                    w = std::min(1.0f, w * (1.0f + annealed_factor * 0.5f));
                                }
                                w_cur[static_cast<std::size_t>(j)] = w;
                            }
                            f_w_out.write(reinterpret_cast<const char*>(w_cur.data()),
                                          static_cast<std::streamsize>(sizeof(float) * w_cur.size()));
                            const DeviceVecF32View d_w = cuda_kernels->KmeansUploadBlockWeights(w_cur);
                            cuda_kernels->KmeansAccumulateAnnealTotalsFromBlockBestDotAndWeights(d_w);
                            cuda_kernels->KmeansAccumulateWeightsDeviceAssignNoAtomic(
                                Xd, cuda_kernels->KmeansBlockAssignView(), d_w);
                        }
                        else {
                            cuda_kernels->KmeansDownloadBlockBestDot(&dot_blk);
#pragma omp simd reduction(+:total_cost)
                            for (int j = 0; j < cols_cur; ++j) {
                                total_cost += static_cast<double>(1.0f - dot_blk[static_cast<std::size_t>(j)]);
                            }
                            total_w += static_cast<double>(cols_cur);
                            cuda_kernels->KmeansAccumulateUnitWeightsDeviceAssignNoAtomic(
                                Xd, cuda_kernels->KmeansBlockAssignView());
                        }
                        TimingComputeEnd(slot_cur);
                        cuda_kernels->KmeansMarkUploadSlotConsumed(slot_cur);

                        cur = next;
                        w_cur = std::move(w_next);
                        std::swap(slot_cur, slot_next);
                    }
                    if (emit_codes_this_iter && cuda_kernels) {
                        // Drain remaining pending slots in order.
                        if (emit_pending[0].pending && emit_pending[1].pending) {
                            const int first = (emit_pending[0].col0 <= emit_pending[1].col0) ? 0 : 1;
                            const int second = 1 - first;
                            if (!EmitFlushSlot(first)) return {};
                            if (!EmitFlushSlot(second)) return {};
                        }
                        else {
                            if (!EmitFlushSlot(0)) return {};
                            if (!EmitFlushSlot(1)) return {};
                        }
                    }
                }
            }
            else {
                int emit_block_i = 0;
                while (true) {
                    if (!ReaderReadNextTimed(block_cols, &blk)) return {};
                    const int cols_blk =
                        (blk.dtype == io::ColBlockDType::kU8) ? blk.X_u8.cols : blk.X_f32.cols;
                    if (cols_blk == 0) break;

                    std::vector<float> w_blk;
                    if (need_weight_files) {
                        ReadVectorExactOrThrow(f_w_in, &w_blk, static_cast<std::size_t>(cols_blk));
                    }

                    if (cuda_kernels) {
                        const int emit_slot = emit_block_i & 1;
                        const DeviceMatF32View Xd = UploadAndNormalizeBlock(cuda_kernels, blk,
                                                                            stage_blocks_to_pinned,
                                                                            profile_timing, timing);
                        TimingComputeBegin(0);
                        if (use_hier2) {
                            Timer t_assign;
                            cuda_kernels->KmeansHier2AssignColsF32Device(Xd, coarse, centers,
                                                                         /*K_valid=*/k,
                                                                         /*K1=*/hier2.K1,
                                                                         /*K2=*/hier2.K2,
                                                                         /*topL=*/cfg.large_k_top_coarse);
                            if (profile_timing && timing) {
                                stream_kernels->SyncCompute();
                                timing->assign_gemm_s += t_assign.ElapsedSeconds();
                            }
                        }
                        else {
                            AssignBlockGpuTiled(Xd, centers, k, stream_kernels, cuda_kernels,
                                                profile_timing, timing,
                                                /*need_host_argmax=*/false,
                                                /*assign_blk=*/nullptr,
                                                /*dot_blk=*/nullptr);
                        }
                        if (!EmitScheduleSlot(emit_slot, blk.col0, cols_blk)) return {};
                        ++emit_block_i;

                        if (cfg.collect_metrics) {
                            cuda_kernels->KmeansAccumulateMseFromBlockBestDot(cols_blk);
                        }
                        if (onepass_active) {
                            cuda_kernels->KmeansAccumulateCostHistogramFromBestDot(
                                cuda_kernels->KmeansBlockBestView(), hist_bins);
                        }
                        if (use_device_weights) {
                            if (adaptive_active) {
                                cuda_kernels->KmeansAnnealUpdateWeightsAndTotalsFromBlockBest(
                                    blk.col0, cols_blk, effective_threshold, annealed_factor, cfg.min_weight);
                            }
                            else {
                                const DeviceVecF32View d_w =
                                    cuda_kernels->KmeansAnnealWeightsBlockView(blk.col0, cols_blk);
                                cuda_kernels->KmeansAccumulateAnnealTotalsFromBlockBestDotAndWeights(d_w);
                            }
                            const DeviceVecF32View d_w =
                                cuda_kernels->KmeansAnnealWeightsBlockView(blk.col0, cols_blk);
                            cuda_kernels->KmeansAccumulateWeightsDeviceAssignNoAtomic(
                                Xd, cuda_kernels->KmeansBlockAssignView(), d_w);
                        }
                        else if (adaptive_active) {
                            cuda_kernels->KmeansComputeOutlierMaskFromBlockBest(cols_blk, effective_threshold);
                            cuda_kernels->KmeansDownloadOutlierMaskAsync(cols_blk);
                            std::uint8_t* mask_ptr = nullptr;
                            int mask_bytes = 0;
                            cuda_kernels->KmeansSyncOutlierMaskDownload(&mask_ptr, &mask_bytes);
                            if (!mask_ptr || mask_bytes < cols_blk) {
                                if (err) *err = "Kmeans outlier mask download failed.";
                                return {};
                            }
                            for (int j = 0; j < cols_blk; ++j) {
                                float w = w_blk[static_cast<std::size_t>(j)];
                                if (mask_ptr[j]) {
                                    w = std::max(cfg.min_weight, w * (1.0f - annealed_factor));
                                }
                                else {
                                    w = std::min(1.0f, w * (1.0f + annealed_factor * 0.5f));
                                }
                                w_blk[static_cast<std::size_t>(j)] = w;
                            }
                            f_w_out.write(reinterpret_cast<const char*>(w_blk.data()),
                                          static_cast<std::streamsize>(sizeof(float) * w_blk.size()));
                            const DeviceVecF32View d_w = cuda_kernels->KmeansUploadBlockWeights(w_blk);
                            cuda_kernels->KmeansAccumulateAnnealTotalsFromBlockBestDotAndWeights(d_w);
                            cuda_kernels->KmeansAccumulateWeightsDeviceAssignNoAtomic(
                                Xd, cuda_kernels->KmeansBlockAssignView(), d_w);
                        }
                        else {
                            cuda_kernels->KmeansDownloadBlockBestDot(&dot_blk);
#pragma omp simd reduction(+:total_cost)
                            for (int j = 0; j < cols_blk; ++j) {
                                total_cost += static_cast<double>(1.0f - dot_blk[static_cast<std::size_t>(j)]);
                            }
                            total_w += static_cast<double>(cols_blk);
                            cuda_kernels->KmeansAccumulateUnitWeightsDeviceAssignNoAtomic(
                                Xd, cuda_kernels->KmeansBlockAssignView());
                        }
                        TimingComputeEnd(0);
                        if (blk.dtype == io::ColBlockDType::kU8) {
                            cuda_kernels->KmeansMarkUploadSlotConsumed(0);
                        }
                        // NOTE: no per-block SyncCompute here; it destroys H2D/compute overlap and can dominate wall time.
                    }
                    else {
                        ColMajorMatrix<float> tmp_f32;
                        ColMajorMatrix<float>* Xblk = nullptr;
                        if (blk.dtype == io::ColBlockDType::kU8) {
                            tmp_f32 = ColMajorMatrix<float>(d, cols_blk);
                            for (int j = 0; j < cols_blk; ++j) {
                                const std::uint8_t* xj = blk.X_u8.Col(j);
                                float* dst = tmp_f32.Col(j);
#pragma omp simd
                                for (int r = 0; r < d; ++r) dst[r] = static_cast<float>(xj[r]);
                            }
                            Xblk = &tmp_f32;
                        }
                        else {
                            Xblk = &blk.X_f32;
                        }
                        NormalizeColumnsInPlace(Xblk);
                        AssignSamplesGpu(*Xblk, centers, stream_kernels, profile_timing, timing,
                                         &assign_blk, &dot_blk);
                        if (adaptive_active) {
                            for (int j = 0; j < cols_blk; ++j) {
                                const float cost = 1.0f - dot_blk[static_cast<std::size_t>(j)];
                                float w = w_blk[static_cast<std::size_t>(j)];
                                if (cost > effective_threshold) {
                                    w = std::max(cfg.min_weight, w * (1.0f - annealed_factor));
                                }
                                else {
                                    w = std::min(1.0f, w * (1.0f + annealed_factor * 0.5f));
                                }
                                w_blk[static_cast<std::size_t>(j)] = w;
                                total_cost += static_cast<double>(w) * static_cast<double>(cost);
                                total_w += static_cast<double>(w);
                            }
                            f_w_out.write(reinterpret_cast<const char*>(w_blk.data()),
                                          static_cast<std::streamsize>(sizeof(float) * w_blk.size()));
                            AccumulateBlockCpu(*Xblk, assign_blk, &w_blk, &sum_x, &sum_w, &offsets, &cursor, &order);
                        }
                        else {
#pragma omp simd reduction(+:total_cost)
                            for (int j = 0; j < cols_blk; ++j) {
                                total_cost += static_cast<double>(1.0f - dot_blk[static_cast<std::size_t>(j)]);
                            }
                            total_w += static_cast<double>(cols_blk);
                            AccumulateBlockCpu(*Xblk, assign_blk, nullptr, &sum_x, &sum_w, &offsets, &cursor, &order);
                        }
                        if (emit_codes_this_iter) {
                            std::string emit_err;
                            if (!emit_codes->WriteBlockFromI32(emit_layer, blk.col0, cols_blk,
                                                               assign_blk.data(), &emit_err)) {
                                if (err)
                                    *err =
                                        "SphericalKmeansCentersOnlyStreamingReader: codes_store.WriteBlockFromI32 failed: "
                                        + emit_err;
                                return {};
                            }
                        }
                    }
                }
                if (emit_codes_this_iter && cuda_kernels) {
                    // Drain any remaining pending slots in order.
                    if (emit_pending[0].pending && emit_pending[1].pending) {
                        const int first = (emit_pending[0].col0 <= emit_pending[1].col0) ? 0 : 1;
                        const int second = 1 - first;
                        if (!EmitFlushSlot(first)) return {};
                        if (!EmitFlushSlot(second)) return {};
                    }
                    else {
                        if (!EmitFlushSlot(0)) return {};
                        if (!EmitFlushSlot(1)) return {};
                    }
                }
            }

            if (need_weight_files) {
                if (!f_w_out) {
                    if (err) *err = "Failed to write weights_out.";
                    return {};
                }
                f_w_out.flush();
                f_w_out.close();
                f_w_in.close();
                std::error_code ec;
                std::filesystem::remove(w_in_path, ec);
                ec.clear();
                std::filesystem::rename(w_out_path, w_in_path, ec);
                if (ec) {
                    if (err) *err = "Failed to rotate weight files: " + ec.message();
                    return {};
                }
            }

            if (cuda_kernels) {
                if (use_device_weights || (adaptive_active && anneal_enabled)) {
                    cuda_kernels->KmeansDownloadAnnealTotalsAsync();
                }
                if (cfg.collect_metrics) {
                    cuda_kernels->KmeansDownloadMetricTotalsAsync();
                }
                cuda_kernels->KmeansDownloadSums(&sum_x, &sum_w);
                cuda_kernels->Sync();
                if (use_device_weights || (adaptive_active && anneal_enabled)) {
                    cuda_kernels->KmeansSyncAnnealTotalsDownload(&total_w, &total_cost);
                }
                if (timing) {
                    if (!cfg.collect_metrics) {
                        timing->mse_proxy = -1.0;
                    }
                    else {
                        double mse_sum = 0.0;
                        double mse_count = 0.0;
                        cuda_kernels->KmeansSyncMetricTotalsDownload(&mse_sum, &mse_count);
                        timing->mse_proxy = (mse_count > 0.0) ? (mse_sum / mse_count) : -1.0;
                    }
                }
            }

            if (onepass_active && scans_this_iter != 1) {
                if (err) {
                    *err = "onepass mode regression: expected 1 dataset scan per adaptive iteration, got " +
                        std::to_string(scans_this_iter);
                }
                return {};
            }

            if (onepass_active) {
                const int bins = std::max(1, cfg.quantile_bins);
                const float lo = 0.0f;
                const float hi = 2.0f;
                const std::uint64_t target = static_cast<std::uint64_t>(
                    cfg.outlier_quantile * static_cast<float>(std::max<std::int64_t>(1, n64 - 1)));
                std::uint64_t prefix = 0;
                int bq = bins - 1;
                std::vector<std::uint32_t> hist_u32(static_cast<std::size_t>(bins));
                cuda_kernels->KmeansDownloadCostHistogramAsync(hist_u32.data(), bins);
                cuda_kernels->KmeansDownloadCostHistogramSync();
                for (int b = 0; b < bins; ++b) {
                    prefix += static_cast<std::uint64_t>(hist_u32[static_cast<std::size_t>(b)]);
                    if (prefix > target) {
                        bq = b;
                        break;
                    }
                }
                const float bw = (hi - lo) / static_cast<float>(bins);
                const float quantile_threshold = lo + (static_cast<float>(bq) + 0.5f) * bw;
                threshold_next = std::max(quantile_threshold, cfg.cost_threshold);
                threshold_prev = threshold_next;
            }

            // Finalize centers from sums.
#pragma omp parallel for default(none) shared(centers, sum_w, fallback, sample_norm, sum_x) firstprivate(d, k, kEps) schedule(static)
            for (int c = 0; c < k; ++c) {
                float* center = centers.Col(c);
                const float w = sum_w[static_cast<std::size_t>(c)];
                if (w <= kEps) {
                    const int idx = fallback[static_cast<std::size_t>(c)];
                    const float* src = sample_norm.Col(idx);
                    std::copy(src, src + d, center);
                    continue;
                }
                const float inv_w = 1.0f / w;
                const float* sx = sum_x.Col(c);
                float norm = 0.0f;
                for (int r = 0; r < d; ++r) {
                    const float v = sx[r] * inv_w;
                    center[r] = v;
                    norm += v * v;
                }
                norm = std::sqrt(std::max(norm, kEps));
                const float inv = 1.0f / norm;
#pragma omp simd
                for (int r = 0; r < d; ++r) {
                    center[r] *= inv;
                }
            }

            const double current_cost = (total_w > 0.0) ? (total_cost / total_w) : 0.0;
            const bool converged = (std::abs(prev_cost - current_cost) < static_cast<double>(cfg.tol));
            if (converged) {
                // Prompt-10: if we need to emit codes but haven't, run exactly one more iteration
                // (armed) so the next iteration emits codes and we can stop without an extra scan.
                if (emit_codes_base_enabled && !emit_codes_done && !emit_codes_this_iter) {
                    emit_codes_armed = true;
                }
                else {
                    if (emit_codes_this_iter) {
                        emit_codes_done = true;
                        emit_codes_armed = false;
                    }
                    break;
                }
            }
            prev_cost = current_cost;
            if (emit_codes_this_iter) {
                emit_codes_done = true;
                emit_codes_armed = false;
            }
        }

        if (profile_timing && timing && cuda_kernels) {
            double upload_h2d_s = 0.0;
            double upload_kernel_s = 0.0;
            double rvq_project_s = 0.0;
            double assign_update_compute_s = 0.0;
            double emit_codes_d2h_s = 0.0;
            double wait_copy_s = 0.0;
            double wait_compute_s = 0.0;
            cuda_kernels->KmeansTimingFlush(&upload_h2d_s, &upload_kernel_s, &rvq_project_s,
                                            &assign_update_compute_s, &emit_codes_d2h_s,
                                            &wait_copy_s, &wait_compute_s);
            timing->normalize_h2d_s += upload_h2d_s;
            timing->normalize_kernel_s += upload_kernel_s;
            timing->rvq_project_s += rvq_project_s;
            timing->assign_update_compute_s += assign_update_compute_s;
            timing->emit_codes_d2h_s += emit_codes_d2h_s;
            timing->wait_copy_s += wait_copy_s;
            timing->wait_compute_s += wait_compute_s;
        }

        if (profile_timing || timing) {
            timing->total_s += t_total.ElapsedSeconds();
        }

        return centers;
    }

    KmeansResult SphericalKmeansInMemory(const ColMajorMatrix<float>& X,
                                         int k,
                                         const KmeansConfig& cfg,
                                         std::mt19937* rng,
                                         StreamKernelProvider* stream_kernels,
                                         bool profile_timing,
                                         KmeansTiming* timing) {
        // Julia reference: spherical_kmeans_fast_with_adaptive_weights (fast_spkmeans.jl:370-469).
        const int d = X.rows;
        const int n = X.cols;

        PopulateKmeansTimingFlags(timing, cfg, k);

        KmeansResult out;
        out.centers = ColMajorMatrix<float>(d, k);
        out.assignments.assign(n, 0);
        out.weights.assign(n, cfg.initial_weight);

        Timer t_total;

        ColMajorMatrix<float> X_norm;
        if (profile_timing && timing) {
            Timer t;
            X_norm = NormalizeColumns(X);
            timing->normalize_s += t.ElapsedSeconds();
        }
        else {
            X_norm = NormalizeColumns(X);
        }

        if (profile_timing && timing) {
            Timer t;
            out.centers = InitializeCentersFast(X_norm, k, cfg, rng);
            const double s = t.ElapsedSeconds();
            timing->init_pick_s += s;
            timing->init_centers_s += s;
        }
        else {
            out.centers = InitializeCentersFast(X_norm, k, cfg, rng);
        }

        std::vector<float> dot_values(n, 0.0f);
        std::vector<float> point_costs;
        std::vector<float> point_costs_copy;
        point_costs.resize(static_cast<std::size_t>(n));
        point_costs_copy.resize(static_cast<std::size_t>(n));
        float prev_cost = std::numeric_limits<float>::infinity();

        for (int iter = 1; iter <= cfg.max_iters; ++iter) {
            // Julia: assign_samples_parallel! (fast_spkmeans.jl:415).
            AssignSamplesGpu(X_norm, out.centers, stream_kernels, profile_timing, timing,
                             &out.assignments, &dot_values);

            // Julia: compute_cost_fast (fast_spkmeans.jl:418-419).
            float current_cost = ComputeCost(dot_values, out.weights);

            if (iter > cfg.warmup_iters) {
                // Julia: adaptive weights (fast_spkmeans.jl:421-442).
                Timer t_w;
                for (int i = 0; i < n; ++i) {
                    point_costs[static_cast<std::size_t>(i)] = 1.0f - dot_values[static_cast<std::size_t>(i)];
                }
                point_costs_copy = point_costs;
                float quantile_threshold = ComputeQuantile(&point_costs_copy, cfg.outlier_quantile);
                float effective_threshold = std::max(quantile_threshold, cfg.cost_threshold);
                float denom = static_cast<float>(std::max(1, cfg.max_iters - cfg.warmup_iters));
                float annealed_factor =
                    cfg.annealing_factor * (1.0f - std::exp(-(iter - cfg.warmup_iters) / denom));
                for (int i = 0; i < n; ++i) {
                    if (point_costs[static_cast<std::size_t>(i)] > effective_threshold) {
                        out.weights[static_cast<std::size_t>(i)] =
                            std::max(cfg.min_weight,
                                     out.weights[static_cast<std::size_t>(i)] * (1.0f - annealed_factor));
                    }
                    else {
                        out.weights[static_cast<std::size_t>(i)] =
                            std::min(1.0f, out.weights[static_cast<std::size_t>(i)] * (1.0f + annealed_factor * 0.5f));
                    }
                }
                if (profile_timing && timing) timing->weights_s += t_w.ElapsedSeconds();
            }

            // Julia: update_centers_fast! (fast_spkmeans.jl:451).
            if (profile_timing && timing) {
                Timer t_upd;
                UpdateCenters(X_norm, out.assignments, out.weights, rng, &out.centers);
                timing->update_s += t_upd.ElapsedSeconds();
            }
            else {
                UpdateCenters(X_norm, out.assignments, out.weights, rng, &out.centers);
            }

            if (std::abs(prev_cost - current_cost) < cfg.tol) {
                break;
            }
            prev_cost = current_cost;
        }

        if (timing) {
            timing->total_s += t_total.ElapsedSeconds();
        }
        return out;
    }

    KmeansResult SphericalKmeans(const ColMajorMatrix<float>& X,
                                 int k,
                                 const KmeansConfig& cfg,
                                 std::mt19937* rng,
                                 StreamKernelProvider* stream_kernels,
                                 bool profile_timing,
                                 KmeansTiming* timing) {
        PopulateKmeansTimingFlags(timing, cfg, k);

        if (!cfg.streaming) {
            return SphericalKmeansInMemory(X, k, cfg, rng, stream_kernels, profile_timing, timing);
        }

        // Streaming mode (blockwise normalize) that avoids materializing X_norm.
        const int d = X.rows;
        const int n = X.cols;
        KmeansResult out;
        out.centers = ColMajorMatrix<float>(d, k);
        out.assignments.assign(static_cast<std::size_t>(n), 0);
        const bool anneal_enabled = AnnealingEnabled(cfg);
        out.weights.clear();

        Timer t_total;

        // Init centers from a normalized random sample.
        if (profile_timing && timing) {
            Timer t;
            ColMajorMatrix<float> sample_norm;
            {
                Timer t_sample;
                sample_norm = BuildNormalizedSample(X, cfg.init_samples, rng);
                timing->init_sample_s += t_sample.ElapsedSeconds();
            }
            {
                Timer t_pick;
                out.centers = InitializeCentersFast(sample_norm, k, cfg, rng);
                timing->init_pick_s += t_pick.ElapsedSeconds();
            }
            timing->init_centers_s += t.ElapsedSeconds();
        }
        else {
            ColMajorMatrix<float> sample_norm = BuildNormalizedSample(X, cfg.init_samples, rng);
            out.centers = InitializeCentersFast(sample_norm, k, cfg, rng);
        }

        const int block_cols = (cfg.block_cols > 0) ? cfg.block_cols : 200000;
        ColMajorMatrix<float> Xblk_norm(d, std::min(n, std::max(1, block_cols)));
        std::vector<int> assign_blk;
        std::vector<float> dot_blk;
        std::vector<float> point_costs;
        if (anneal_enabled) {
            // Keep empty; large-N streaming will spill per-sample costs/weights to disk.
            point_costs.clear();
        }

        std::vector<int> fallback(static_cast<std::size_t>(k), 0);
        {
            std::uniform_int_distribution<int> uni(0, std::max(0, n - 1));
            for (int c = 0; c < k; ++c) fallback[static_cast<std::size_t>(c)] = uni(*rng);
        }

        CudaStreamKernels* cuda_kernels = nullptr;
        if (stream_kernels && stream_kernels->IsGpu()) {
            cuda_kernels = dynamic_cast<CudaStreamKernels*>(stream_kernels);
        }
        if (cuda_kernels) {
            cuda_kernels->KmeansSetCollectMetrics(cfg.collect_metrics);
        }

        // Device-resident anneal weights (best-effort). In this in-memory streaming path, the
        // legacy fallback uses on-disk weight files; when device weights are available we avoid
        // creating any `kmeans_weights_*.bin` files.
        const bool request_onepass = (cfg.anneal_mode == KmeansAnnealMode::kOnePass);
        const bool want_device_weights =
            (cuda_kernels != nullptr) && cfg.anneal_no_spill && cfg.anneal_weights_device && !cfg.
            force_disable_device_weights;
        const bool use_device_weights =
            (anneal_enabled && want_device_weights)
                ? cuda_kernels->KmeansInitAnnealWeightsAll(n, cfg.initial_weight)
                : false;
        const bool onepass_enabled = request_onepass && use_device_weights;
        static bool warned_onepass_fallback = false;
        if (request_onepass && !onepass_enabled && anneal_enabled && !warned_onepass_fallback) {
            warned_onepass_fallback = true;
            LogWarn(
                "kmeans anneal_mode=onepass requested, but device weights unavailable; falling back to twopass/file weights.");
        }
        static bool warned_device_weights_oom = false;
        if (anneal_enabled && want_device_weights && !use_device_weights && !warned_device_weights_oom) {
            warned_device_weights_oom = true;
            LogWarn("kmeans device-resident weights unavailable (OOM or disabled); using file weights fallback.");
        }

        // Streaming+annealing fallback (file weights) uses temp files; initialize only when needed.
        if (anneal_enabled && !use_device_weights) {
            const std::string tmp_dir = !cfg.tmp_dir.empty() ? cfg.tmp_dir : std::string("./tmp");
            std::filesystem::create_directories(tmp_dir);
            const std::string costs_path = tmp_dir + "/kmeans_costs_f32.bin";
            const std::string assign_u16_path = tmp_dir + "/kmeans_assign_u16.bin";
            const std::string assign_u32_path = tmp_dir + "/kmeans_assign_u32.bin";
            const std::string w_in_path = tmp_dir + "/kmeans_weights_in_f32.bin";
            const std::string w_out_path = tmp_dir + "/kmeans_weights_out_f32.bin";
            std::error_code ec;
            std::filesystem::remove(costs_path, ec);
            std::filesystem::remove(assign_u16_path, ec);
            std::filesystem::remove(assign_u32_path, ec);
            std::filesystem::remove(w_out_path, ec);
            // Always overwrite weights_in so stale runs never affect current training.
            WriteConstantF32VectorOrThrow(w_in_path, static_cast<std::size_t>(n), cfg.initial_weight);
        }

        // Optional: treat the (host) input as uint8-valued for the purpose of GPU upload+normalize
        // to reduce H2D bandwidth. This must only be enabled for layer-0 (raw SIFT1B vectors),
        // and is disabled by callers for residual layers.
        const bool use_u8_upload = (cuda_kernels != nullptr) && cfg.bvecs_use_u8;
        std::vector<std::uint8_t> X_u8;
        const std::uint8_t* X_u8_base = nullptr;
        if (use_u8_upload) {
            Timer t_pack;
            X_u8.resize(static_cast<std::size_t>(d) * static_cast<std::size_t>(n));
#pragma omp parallel for default(none) shared(X, X_u8) firstprivate(d, n) schedule(static)
            for (int j = 0; j < n; ++j) {
                const float* src = X.Col(j);
                std::uint8_t* dst = X_u8.data() + static_cast<std::size_t>(j) * static_cast<std::size_t>(d);
                // NOTE: we assume src[r] is already an exact uint8 value in [0,255] (SIFT1B raw vectors).
                // No clamp/round here to preserve exactness and avoid extra branches.
#pragma omp simd
                for (int r = 0; r < d; ++r) {
                    dst[r] = static_cast<std::uint8_t>(src[r]);
                }
            }
            X_u8_base = X_u8.data();
            if (profile_timing && timing) {
                // Attribute packing to "other" via total_s - sum(stages); we still record it in normalize_s
                // because it's part of the "make block upload-ready" stage.
                timing->normalize_stage_s += t_pack.ElapsedSeconds();
            }
        }
        // Best-effort: cache the full normalized X on device once, and reuse it across iterations.
        // This mirrors the "full upload" behavior when X fits on GPU, but remains valid for streaming
        // (falls back automatically if memory is insufficient).
        DeviceMatF32View Xnorm_full;
        bool have_full_xnorm = false;
        const float* X_pinned_base = nullptr;
        if (!use_u8_upload && cuda_kernels && cfg.cache_xnorm_device) {
            Timer t_cache;
            have_full_xnorm = cuda_kernels->KmeansTryCacheXnormOnDevice(X, &Xnorm_full);
            if (!have_full_xnorm) {
                if (cfg.pin_host_x) {
                    (void)cuda_kernels->KmeansTryPinHostX(X, &X_pinned_base);
                }
            }
            if (have_full_xnorm && profile_timing && timing) {
                timing->normalize_s += t_cache.ElapsedSeconds();
            }
        }
        else if (!use_u8_upload && cuda_kernels && cfg.pin_host_x) {
            (void)cuda_kernels->KmeansTryPinHostX(X, &X_pinned_base);
        }
        if (timing) {
            timing->used_device_xnorm_cache = have_full_xnorm ? 1 : 0;
            // For u8 uploads, we stage into pinned buffers when pin_host_x is true.
            timing->used_pinned_host_x = (
                use_u8_upload ? (cfg.pin_host_x ? 1 : 0) : (X_pinned_base != nullptr ? 1 : 0));
        }

        float threshold_prev = cfg.cost_threshold;
        float prev_cost = std::numeric_limits<float>::infinity();
        for (int iter = 1; iter <= cfg.max_iters; ++iter) {
            const bool adaptive_active = (anneal_enabled && iter > cfg.warmup_iters);
            const bool onepass_active = adaptive_active && onepass_enabled;
            const bool twopass_active = adaptive_active && !onepass_active;
            const bool use_hier2 =
                (cuda_kernels != nullptr) && cfg.large_k_hier2_train && ShouldUseHier2(k, cfg);
            Hier2Split hier2;
            ColMajorMatrix<float> coarse;
            if (use_hier2) {
                hier2 = ComputeHier2Split(k, cfg);
                coarse = BuildCoarseFromFineCPU(out.centers, k, hier2.K1, hier2.K2);
                if (timing) {
                    timing->used_hier2 = 1;
                    timing->hier2_K = std::max(timing->hier2_K, k);
                    timing->hier2_K1 = hier2.K1;
                    timing->hier2_K2 = hier2.K2;
                }
            }

            // One-pass path (no adaptive weights): assignment + accumulation in a single scan.
            if (!adaptive_active) {
                ColMajorMatrix<float> sum_x(d, k);
                std::vector<float> sum_w(static_cast<std::size_t>(k), 0.0f);

                double total_cost = 0.0;
                double total_w = 0.0;

                double acc_s = 0.0;
                // Reuse bucket scratch buffers across blocks (avoid repeated allocations).
                std::vector<int> offsets(static_cast<std::size_t>(k) + 1, 0);
                std::vector<int> cursor(static_cast<std::size_t>(k) + 1, 0);
                std::vector<int> order;

                if (cuda_kernels) {
                    cuda_kernels->KmeansResetSums(d, k);
                }

                const bool overlap_h2d =
                    (cuda_kernels != nullptr) && (!have_full_xnorm) && (block_cols >= 4096);
                if (timing && overlap_h2d) {
                    // One-pass streaming: this is the main assignment/update scan.
                    timing->used_overlap_pass2 = 1;
                }
                int overlap_slot_cur = 0;
                int overlap_slot_next = 1;
                DeviceMatF32View overlap_Xd[2];
                auto ScheduleUpload = [&](int slot, int i0, int cols_upload) -> DeviceMatF32View
                {
                    if (use_u8_upload) {
                        const std::uint8_t* xptr =
                            X_u8_base + static_cast<std::size_t>(i0) * static_cast<std::size_t>(d);
                        return cuda_kernels->KmeansUploadU8AndNormalizeHostPtrBAsync(
                            slot,
                            xptr, /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols_upload,
                            /*stage_to_pinned=*/cfg.pin_host_x,
                            /*stage_s=*/nullptr,
                            /*h2d_s=*/nullptr,
                            /*kernel_s=*/nullptr);
                    }
                    else {
                        const float* xptr =
                            (X_pinned_base ? X_pinned_base : X.data.data()) +
                            static_cast<std::size_t>(i0) * static_cast<std::size_t>(d);
                        if (X_pinned_base) {
                            return cuda_kernels->KmeansUploadAndNormalizeHostPtrBAsync(
                                slot,
                                xptr, /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols_upload,
                                /*stage_to_pinned=*/false,
                                /*stage_s=*/nullptr,
                                /*h2d_s=*/nullptr,
                                /*kernel_s=*/nullptr);
                        }
                        float* pinned = cuda_kernels->KmeansGetPinnedStageBuffer(slot, d, cols_upload);
                        std::memcpy(pinned,
                                    xptr,
                                    sizeof(float) * static_cast<std::size_t>(d) * static_cast<std::size_t>(
                                        cols_upload));
                        return cuda_kernels->KmeansUploadAndNormalizePinnedAsync(
                            slot,
                            pinned, /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols_upload,
                            /*h2d_s=*/nullptr,
                            /*kernel_s=*/nullptr);
                    }
                };
                if (overlap_h2d && n > 0) {
                    const int cols0 = std::min(block_cols, n);
                    overlap_slot_cur = 0;
                    overlap_slot_next = 1;
                    overlap_Xd[overlap_slot_cur] = ScheduleUpload(overlap_slot_cur, /*i0=*/0, cols0);
                }

                for (int i0 = 0; i0 < n; i0 += block_cols) {
                    const int cols = std::min(block_cols, n - i0);
                    DeviceMatF32View Xd;
                    if (cuda_kernels) {
                        if (have_full_xnorm && Xnorm_full.ptr && Xnorm_full.rows == d && Xnorm_full.cols == n) {
                            Xd.ptr = Xnorm_full.ptr +
                                static_cast<std::size_t>(i0) * static_cast<std::size_t>(Xnorm_full.ld);
                            Xd.rows = d;
                            Xd.cols = cols;
                            Xd.ld = Xnorm_full.ld;
                        }
                        else {
                            if (overlap_h2d) {
                                cuda_kernels->KmeansComputeWaitForUpload(overlap_slot_cur);
                                Xd = overlap_Xd[overlap_slot_cur];
                            }
                            else {
                                Timer t_norm;
                                const float* xptr_f32 = (X_pinned_base ? X_pinned_base : X.data.data()) +
                                    static_cast<std::size_t>(i0) * static_cast<std::size_t>(d);
                                const std::uint8_t* xptr_u8 = X_u8_base
                                                                  ? (X_u8_base + static_cast<std::size_t>(i0) *
                                                                      static_cast<std::size_t>(d))
                                                                  : nullptr;
                                double stage_s = 0.0;
                                double h2d_s = 0.0;
                                double ker_s = 0.0;
                                if (profile_timing && timing) {
                                    if (use_u8_upload) {
                                        Xd = cuda_kernels->KmeansUploadU8AndNormalizeHostPtrBAsync(
                                            /*slot=*/0,
                                                     xptr_u8, /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols,
                                                     /*stage_to_pinned=*/cfg.pin_host_x,
                                                     /*stage_s=*/&stage_s, &h2d_s, &ker_s);
                                        cuda_kernels->KmeansComputeWaitForUpload(0);
                                    }
                                    else {
                                        Xd = cuda_kernels->KmeansUploadAndNormalizeHostPtrBTimed(
                                            xptr_f32, /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols,
                                            /*stage_to_pinned=*/(X_pinned_base == nullptr),
                                            /*stage_s=*/&stage_s, &h2d_s, &ker_s);
                                    }
                                }
                                else {
                                    if (use_u8_upload) {
                                        Xd = cuda_kernels->KmeansUploadU8AndNormalizeHostPtrBAsync(
                                            /*slot=*/0,
                                                     xptr_u8, /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols,
                                                     /*stage_to_pinned=*/cfg.pin_host_x,
                                                     /*stage_s=*/nullptr,
                                                     /*h2d_s=*/nullptr,
                                                     /*kernel_s=*/nullptr);
                                        cuda_kernels->KmeansComputeWaitForUpload(0);
                                    }
                                    else {
                                        Xd = cuda_kernels->KmeansUploadAndNormalizeHostPtrBTimed(
                                            xptr_f32, /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols,
                                            /*stage_to_pinned=*/(X_pinned_base == nullptr),
                                            /*stage_s=*/nullptr,
                                            /*h2d_s=*/nullptr,
                                            /*kernel_s=*/nullptr);
                                    }
                                }
                                if (profile_timing && timing) {
                                    timing->normalize_stage_s += stage_s;
                                    timing->normalize_h2d_s += h2d_s;
                                    timing->normalize_kernel_s += ker_s;
                                    timing->normalize_s += t_norm.ElapsedSeconds();
                                }
                            }
                        }

                        // IMPORTANT: do not form the full k x cols score matrix when cols is large.
                        // Tile over columns to bound GPU memory usage (same idea as AssignSamplesGpu).
                        if (use_hier2) {
                            Timer t_assign;
                            cuda_kernels->KmeansHier2AssignColsF32Device(Xd, coarse, out.centers,
                                                                         /*K_valid=*/k,
                                                                         /*K1=*/hier2.K1,
                                                                         /*K2=*/hier2.K2,
                                                                         /*topL=*/cfg.large_k_top_coarse);
                            if (profile_timing && timing) {
                                stream_kernels->SyncCompute();
                                // Hier2 assignment includes a coarse GEMM + selection + fine scan.
                                // We attribute the full cost to `assign_gemm_s` for now.
                                timing->assign_gemm_s += t_assign.ElapsedSeconds();
                            }
                        }
                        else {
                            constexpr std::size_t kScoreBudgetBytes = 256ull * 1024ull * 1024ull; // 256 MiB
                            int tile_n = static_cast<int>(kScoreBudgetBytes /
                                (sizeof(float) * static_cast<std::size_t>(std::max(1, k))));
                            tile_n = std::max(1, tile_n);
                            tile_n = std::min(tile_n, cols);
                            if (tile_n > 256) tile_n = (tile_n / 256) * 256;
                            tile_n = std::max(1, tile_n);

                            cuda_kernels->KmeansEnsureBlockArgmax(cols);
                            DeviceVecI32View d_assign_tile;
                            DeviceVecF32View d_best_tile;
                            for (int j0 = 0; j0 < cols; j0 += tile_n) {
                                const int tcols = std::min(tile_n, cols - j0);
                                const float* dX =
                                    Xd.ptr + static_cast<std::size_t>(j0) * static_cast<std::size_t>(Xd.ld);

                                DeviceMatF32View scores;
                                Timer t_gemm;
                                stream_kernels->GemmDeviceDevicePtrB(/*transA=*/true, /*transB=*/false,
                                    /*alpha=*/1.0f,
                                    out.centers,
                                    dX,
                                    /*ldB=*/Xd.ld,
                                    /*rowsB=*/Xd.rows,
                                    /*colsB=*/tcols,
                                    /*beta=*/0.0f,
                                    &scores);
                                if (profile_timing && timing) {
                                    stream_kernels->SyncCompute();
                                    timing->assign_gemm_s += t_gemm.ElapsedSeconds();
                                }

                                Timer t_arg;
                                stream_kernels->ArgmaxColsF32Device(scores, &d_assign_tile, &d_best_tile);
                                cuda_kernels->KmeansCopyTileArgmaxToBlock(d_assign_tile, d_best_tile, j0);
                                if (profile_timing && timing) {
                                    stream_kernels->SyncCompute();
                                    timing->assign_argmax_s += t_arg.ElapsedSeconds();
                                }
                            }
                        }
                        if (overlap_h2d) {
                            const int i0_next = i0 + block_cols;
                            if (i0_next < n) {
                                const int cols_next = std::min(block_cols, n - i0_next);
                                overlap_Xd[overlap_slot_next] =
                                    ScheduleUpload(overlap_slot_next, i0_next, cols_next);
                            }
                        }
                        cuda_kernels->KmeansDownloadBlockArgmax(&assign_blk, &dot_blk);
                        if (use_u8_upload && !overlap_h2d) {
                            cuda_kernels->KmeansMarkUploadSlotConsumed(0);
                        }
                    }
                    else {
                        if (Xblk_norm.cols != cols) {
                            Xblk_norm = ColMajorMatrix<float>(d, cols);
                        }

                        if (profile_timing && timing) {
                            Timer t_norm;
#pragma omp parallel for default(none) shared(X, Xblk_norm) firstprivate(i0, cols, d, kEps) schedule(static)
                            for (int j = 0; j < cols; ++j) {
                                const float* xi = X.Col(i0 + j);
                                float* dst = Xblk_norm.Col(j);
                                float norm = 0.0f;
#pragma omp simd reduction(+:norm)
                                for (int r = 0; r < d; ++r) norm += xi[r] * xi[r];
                                norm = std::sqrt(std::max(norm, kEps));
                                const float inv = (norm > kEps) ? (1.0f / norm) : 1.0f;
#pragma omp simd
                                for (int r = 0; r < d; ++r) dst[r] = xi[r] * inv;
                            }
                            timing->normalize_s += t_norm.ElapsedSeconds();
                        }
                        else {
#pragma omp parallel for default(none) shared(X, Xblk_norm) firstprivate(i0, cols, d, kEps) schedule(static)
                            for (int j = 0; j < cols; ++j) {
                                const float* xi = X.Col(i0 + j);
                                float* dst = Xblk_norm.Col(j);
                                float norm = 0.0f;
#pragma omp simd reduction(+:norm)
                                for (int r = 0; r < d; ++r) norm += xi[r] * xi[r];
                                norm = std::sqrt(std::max(norm, kEps));
                                const float inv = (norm > kEps) ? (1.0f / norm) : 1.0f;
#pragma omp simd
                                for (int r = 0; r < d; ++r) dst[r] = xi[r] * inv;
                            }
                        }

                        AssignSamplesGpu(Xblk_norm, out.centers, stream_kernels,
                                         profile_timing, timing,
                                         &assign_blk, &dot_blk);
                    }
#pragma omp simd reduction(+:total_cost)
                    for (int j = 0; j < cols; ++j) {
                        out.assignments[static_cast<std::size_t>(i0 + j)] =
                            assign_blk[static_cast<std::size_t>(j)];
                        total_cost += static_cast<double>(1.0f - dot_blk[static_cast<std::size_t>(j)]);
                    }
                    total_w += static_cast<double>(cols);

                    if (cuda_kernels) {
                        Timer t_acc;
                        cuda_kernels->KmeansAccumulateUnitWeightsDeviceAssignNoAtomic(
                            Xd, cuda_kernels->KmeansBlockAssignView());
                        if (profile_timing && timing) {
                            cuda_kernels->SyncCompute();
                        }
                        acc_s += t_acc.ElapsedSeconds();
                    }
                    else {
                        Timer t_acc;
                        // Bucket within this block (per-cluster contiguous ranges).
                        std::fill(offsets.begin(), offsets.end(), 0);
                        for (int j = 0; j < cols; ++j) {
                            const int c = assign_blk[static_cast<std::size_t>(j)];
                            ++offsets[static_cast<std::size_t>(c) + 1];
                        }
                        for (int c = 0; c < k; ++c) {
                            offsets[static_cast<std::size_t>(c) + 1] += offsets[static_cast<std::size_t>(c)];
                        }
                        order.resize(static_cast<std::size_t>(cols));
                        std::copy(offsets.begin(), offsets.end(), cursor.begin());
                        for (int j = 0; j < cols; ++j) {
                            const int c = assign_blk[static_cast<std::size_t>(j)];
                            const int pos = cursor[static_cast<std::size_t>(c)]++;
                            order[static_cast<std::size_t>(pos)] = j;
                        }

#pragma omp parallel default(none) shared(offsets, order, Xblk_norm, sum_x, sum_w) firstprivate(d, k)
                        {
                            std::vector<float> local_sum(static_cast<std::size_t>(d));
#pragma omp for schedule(static)
                            for (int c = 0; c < k; ++c) {
                                const int begin = offsets[static_cast<std::size_t>(c)];
                                const int end = offsets[static_cast<std::size_t>(c) + 1];
                                if (begin == end) continue;

                                std::fill(local_sum.begin(), local_sum.end(), 0.0f);
                                for (int p = begin; p < end; ++p) {
                                    const int j = order[static_cast<std::size_t>(p)];
                                    const float* xj = Xblk_norm.Col(j);
#pragma omp simd
                                    for (int r = 0; r < d; ++r) {
                                        local_sum[static_cast<std::size_t>(r)] += xj[r];
                                    }
                                }
                                float* dst = sum_x.Col(c);
#pragma omp simd
                                for (int r = 0; r < d; ++r) {
                                    dst[r] += local_sum[static_cast<std::size_t>(r)];
                                }
                                sum_w[static_cast<std::size_t>(c)] += static_cast<float>(end - begin);
                            }
                        }
                        acc_s += t_acc.ElapsedSeconds();
                    }
                    if (overlap_h2d) {
                        // Mark the current upload slot as consumed so the copy stream won't reuse it
                        // while compute is still reading Xd (H2D/normalize overlap safety).
                        cuda_kernels->KmeansMarkUploadSlotConsumed(overlap_slot_cur);
                        std::swap(overlap_slot_cur, overlap_slot_next);
                    }
                }

                const float current_cost =
                    (total_w > 0.0) ? static_cast<float>(total_cost / total_w) : 0.0f;

                if (cuda_kernels) {
                    cuda_kernels->KmeansDownloadSums(&sum_x, &sum_w);
                    cuda_kernels->Sync();
                }

                // Finalize centers from sums.
                Timer t_fin;
#pragma omp parallel for default(none) shared(out, sum_w, fallback, X, sum_x) firstprivate(d, k, kEps) schedule(static)
                for (int c = 0; c < k; ++c) {
                    float* center = out.centers.Col(c);
                    const float w = sum_w[static_cast<std::size_t>(c)];
                    if (w <= kEps) {
                        const int idx = fallback[static_cast<std::size_t>(c)];
                        const float* xi = X.Col(idx);
                        float norm = 0.0f;
#pragma omp simd reduction(+:norm)
                        for (int r = 0; r < d; ++r) norm += xi[r] * xi[r];
                        norm = std::sqrt(std::max(norm, kEps));
                        const float inv = (norm > kEps) ? (1.0f / norm) : 1.0f;
#pragma omp simd
                        for (int r = 0; r < d; ++r) center[r] = xi[r] * inv;
                        continue;
                    }
                    const float inv_w = 1.0f / w;
                    const float* sx = sum_x.Col(c);
                    float norm = 0.0f;
                    for (int r = 0; r < d; ++r) {
                        const float v = sx[r] * inv_w;
                        center[r] = v;
                        norm += v * v;
                    }
                    norm = std::sqrt(std::max(norm, kEps));
                    const float inv = 1.0f / norm;
#pragma omp simd
                    for (int r = 0; r < d; ++r) center[r] *= inv;
                }
                const double fin_s = t_fin.ElapsedSeconds();
                if (profile_timing && timing) {
                    timing->accumulate_s += acc_s;
                    timing->finalize_s += fin_s;
                    timing->update_s += (acc_s + fin_s);
                }

                if (std::abs(prev_cost - current_cost) < cfg.tol) {
                    break;
                }
                prev_cost = current_cost;
                continue;
            }

            // Two-pass path (adaptive weights): Pass 1 assignment/costs -> update weights -> Pass 2 update centers.
            // For large-N streaming: avoid materializing point_costs/weights in RAM.
            // We spill per-sample (assignment, cost, weight) to disk in `cfg.tmp_dir`.
            if (cfg.anneal_no_spill && cuda_kernels && use_device_weights) {
                // Device-resident weights path: avoid any per-sample spill and avoid `kmeans_weights_*.bin`.
                const int bins = std::max(1, cfg.quantile_bins);
                const float lo = 0.0f;
                const float hi = 2.0f;

                const float denom = static_cast<float>(std::max(1, cfg.max_iters - cfg.warmup_iters));
                const float annealed_factor =
                    cfg.annealing_factor * (1.0f - std::exp(-(iter - cfg.warmup_iters) / denom));

                float effective_threshold = cfg.cost_threshold;
                if (twopass_active) {
                    cuda_kernels->KmeansResetCostHistogram(bins);

                    const bool overlap_h2d_pass1 =
                        (cuda_kernels != nullptr) && (!have_full_xnorm) && (block_cols >= 4096);
                    if (timing && overlap_h2d_pass1) timing->used_overlap_pass1 = 1;
                    int overlap_slot_cur_pass1 = 0;
                    int overlap_slot_next_pass1 = 1;
                    DeviceMatF32View overlap_Xd_pass1[2];
                    auto ScheduleUploadPass1 = [&](int slot, int i0, int cols_upload) -> DeviceMatF32View
                    {
                        if (use_u8_upload) {
                            const std::uint8_t* xptr =
                                X_u8_base + static_cast<std::size_t>(i0) * static_cast<std::size_t>(d);
                            return cuda_kernels->KmeansUploadU8AndNormalizeHostPtrBAsync(
                                slot,
                                xptr, /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols_upload,
                                /*stage_to_pinned=*/cfg.pin_host_x,
                                /*stage_s=*/nullptr,
                                /*h2d_s=*/nullptr,
                                /*kernel_s=*/nullptr);
                        }
                        const float* xptr =
                            (X_pinned_base ? X_pinned_base : X.data.data()) +
                            static_cast<std::size_t>(i0) * static_cast<std::size_t>(d);
                        if (X_pinned_base) {
                            return cuda_kernels->KmeansUploadAndNormalizeHostPtrBAsync(
                                slot,
                                xptr, /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols_upload,
                                /*stage_to_pinned=*/false,
                                /*stage_s=*/nullptr,
                                /*h2d_s=*/nullptr,
                                /*kernel_s=*/nullptr);
                        }
                        float* pinned = cuda_kernels->KmeansGetPinnedStageBuffer(slot, d, cols_upload);
                        std::memcpy(pinned,
                                    xptr,
                                    sizeof(float) * static_cast<std::size_t>(d) * static_cast<std::size_t>(
                                        cols_upload));
                        return cuda_kernels->KmeansUploadAndNormalizePinnedAsync(
                            slot,
                            pinned, /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols_upload,
                            /*h2d_s=*/nullptr,
                            /*kernel_s=*/nullptr);
                    };
                    if (overlap_h2d_pass1 && n > 0) {
                        const int cols0 = std::min(block_cols, n);
                        overlap_Xd_pass1[overlap_slot_cur_pass1] =
                            ScheduleUploadPass1(overlap_slot_cur_pass1, 0, cols0);
                    }
                    for (int i0 = 0; i0 < n; i0 += block_cols) {
                        const int cols = std::min(block_cols, n - i0);
                        DeviceMatF32View Xd;
                        if (have_full_xnorm && Xnorm_full.ptr && Xnorm_full.rows == d && Xnorm_full.cols == n) {
                            Xd.ptr = Xnorm_full.ptr +
                                static_cast<std::size_t>(i0) * static_cast<std::size_t>(Xnorm_full.ld);
                            Xd.rows = d;
                            Xd.cols = cols;
                            Xd.ld = Xnorm_full.ld;
                        }
                        else if (overlap_h2d_pass1) {
                            cuda_kernels->KmeansComputeWaitForUpload(overlap_slot_cur_pass1);
                            Xd = overlap_Xd_pass1[overlap_slot_cur_pass1];
                        }
                        else {
                            const float* xptr_f32 = (X_pinned_base ? X_pinned_base : X.data.data()) +
                                static_cast<std::size_t>(i0) * static_cast<std::size_t>(d);
                            const std::uint8_t* xptr_u8 = X_u8_base
                                                              ? (X_u8_base + static_cast<std::size_t>(i0) *
                                                                  static_cast<std::size_t>(d))
                                                              : nullptr;
                            if (use_u8_upload) {
                                Xd = cuda_kernels->KmeansUploadU8AndNormalizeHostPtrBAsync(
                                    /*slot=*/0,
                                             xptr_u8, /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols,
                                             /*stage_to_pinned=*/cfg.pin_host_x,
                                             /*stage_s=*/nullptr,
                                             /*h2d_s=*/nullptr,
                                             /*kernel_s=*/nullptr);
                                cuda_kernels->KmeansComputeWaitForUpload(0);
                            }
                            else {
                                Xd = cuda_kernels->KmeansUploadAndNormalizeHostPtrBTimed(
                                    xptr_f32, /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols,
                                    /*stage_to_pinned=*/(X_pinned_base == nullptr),
                                    /*stage_s=*/nullptr,
                                    /*h2d_s=*/nullptr,
                                    /*kernel_s=*/nullptr);
                            }
                        }

                        if (use_hier2) {
                            cuda_kernels->KmeansHier2AssignColsF32Device(Xd, coarse, out.centers,
                                                                         /*K_valid=*/k,
                                                                         /*K1=*/hier2.K1,
                                                                         /*K2=*/hier2.K2,
                                                                         /*topL=*/cfg.large_k_top_coarse);
                        }
                        else {
                            constexpr std::size_t kScoreBudgetBytes = 256ull * 1024ull * 1024ull;
                            int tile_n = static_cast<int>(kScoreBudgetBytes /
                                (static_cast<std::size_t>(k) * sizeof(float)));
                            tile_n = std::min(tile_n, cols);
                            if (tile_n > 256) tile_n = (tile_n / 256) * 256;
                            tile_n = std::max(1, tile_n);

                            cuda_kernels->KmeansEnsureBlockArgmax(cols);
                            DeviceVecI32View d_assign_tile;
                            DeviceVecF32View d_best_tile;
                            for (int j0 = 0; j0 < cols; j0 += tile_n) {
                                const int tcols = std::min(tile_n, cols - j0);
                                const float* dX =
                                    Xd.ptr + static_cast<std::size_t>(j0) * static_cast<std::size_t>(Xd.ld);
                                DeviceMatF32View scores;
                                stream_kernels->GemmDeviceDevicePtrB(/*transA=*/true, /*transB=*/false,
                                    /*alpha=*/1.0f,
                                    out.centers,
                                    dX,
                                    /*ldB=*/Xd.ld,
                                    /*rowsB=*/Xd.rows,
                                    /*colsB=*/tcols,
                                    /*beta=*/0.0f,
                                    &scores);
                                stream_kernels->ArgmaxColsF32Device(scores, &d_assign_tile, &d_best_tile);
                                cuda_kernels->KmeansCopyTileArgmaxToBlock(d_assign_tile, d_best_tile, j0);
                            }
                        }
                        if (overlap_h2d_pass1) {
                            const int i0_next = i0 + block_cols;
                            if (i0_next < n) {
                                const int cols_next = std::min(block_cols, n - i0_next);
                                overlap_Xd_pass1[overlap_slot_next_pass1] =
                                    ScheduleUploadPass1(overlap_slot_next_pass1, i0_next, cols_next);
                            }
                        }
                        cuda_kernels->KmeansAccumulateCostHistogramFromBestDot(
                            cuda_kernels->KmeansBlockBestView(), bins);
                        if (use_u8_upload && !overlap_h2d_pass1) {
                            cuda_kernels->KmeansMarkUploadSlotConsumed(0);
                        }
                        if (overlap_h2d_pass1) {
                            cuda_kernels->KmeansMarkUploadSlotConsumed(overlap_slot_cur_pass1);
                            std::swap(overlap_slot_cur_pass1, overlap_slot_next_pass1);
                        }
                    }

                    std::vector<std::uint32_t> hist_u32(static_cast<std::size_t>(bins));
                    cuda_kernels->KmeansDownloadCostHistogramAsync(hist_u32.data(), bins);
                    cuda_kernels->KmeansDownloadCostHistogramSync();
                    const std::uint64_t target = static_cast<std::uint64_t>(
                        cfg.outlier_quantile * static_cast<float>(std::max(1, n - 1)));
                    std::uint64_t prefix = 0;
                    int bq = bins - 1;
                    for (int b = 0; b < bins; ++b) {
                        prefix += static_cast<std::uint64_t>(hist_u32[static_cast<std::size_t>(b)]);
                        if (prefix > target) {
                            bq = b;
                            break;
                        }
                    }
                    const float bw = (hi - lo) / static_cast<float>(bins);
                    const float quantile_threshold = lo + (static_cast<float>(bq) + 0.5f) * bw;
                    effective_threshold = std::max(quantile_threshold, cfg.cost_threshold);
                }
                else {
                    effective_threshold = threshold_prev;
                }

                ColMajorMatrix<float> sum_x(d, k);
                std::vector<float> sum_w(static_cast<std::size_t>(k), 0.0f);
                cuda_kernels->KmeansResetSums(d, k);
                if (cfg.collect_metrics) {
                    cuda_kernels->KmeansResetMetricTotals();
                }
                cuda_kernels->KmeansResetAnnealTotals();
                if (onepass_active) {
                    cuda_kernels->KmeansResetCostHistogram(bins);
                }

                const bool overlap_h2d_pass2 =
                    (cuda_kernels != nullptr) && (!have_full_xnorm) && (block_cols >= 4096);
                if (timing && overlap_h2d_pass2) timing->used_overlap_pass2 = 1;
                int overlap_slot_cur_pass2 = 0;
                int overlap_slot_next_pass2 = 1;
                DeviceMatF32View overlap_Xd_pass2[2];
                auto ScheduleUploadPass2 = [&](int slot, int i0, int cols_upload) -> DeviceMatF32View
                {
                    if (use_u8_upload) {
                        const std::uint8_t* xptr =
                            X_u8_base + static_cast<std::size_t>(i0) * static_cast<std::size_t>(d);
                        return cuda_kernels->KmeansUploadU8AndNormalizeHostPtrBAsync(
                            slot,
                            xptr, /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols_upload,
                            /*stage_to_pinned=*/cfg.pin_host_x,
                            /*stage_s=*/nullptr,
                            /*h2d_s=*/nullptr,
                            /*kernel_s=*/nullptr);
                    }
                    const float* xptr =
                        (X_pinned_base ? X_pinned_base : X.data.data()) +
                        static_cast<std::size_t>(i0) * static_cast<std::size_t>(d);
                    if (X_pinned_base) {
                        return cuda_kernels->KmeansUploadAndNormalizeHostPtrBAsync(
                            slot,
                            xptr, /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols_upload,
                            /*stage_to_pinned=*/false,
                            /*stage_s=*/nullptr,
                            /*h2d_s=*/nullptr,
                            /*kernel_s=*/nullptr);
                    }
                    float* pinned = cuda_kernels->KmeansGetPinnedStageBuffer(slot, d, cols_upload);
                    std::memcpy(pinned,
                                xptr,
                                sizeof(float) * static_cast<std::size_t>(d) * static_cast<std::size_t>(cols_upload));
                    return cuda_kernels->KmeansUploadAndNormalizePinnedAsync(
                        slot,
                        pinned, /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols_upload,
                        /*h2d_s=*/nullptr,
                        /*kernel_s=*/nullptr);
                };
                if (overlap_h2d_pass2 && n > 0) {
                    const int cols0 = std::min(block_cols, n);
                    overlap_Xd_pass2[overlap_slot_cur_pass2] =
                        ScheduleUploadPass2(overlap_slot_cur_pass2, 0, cols0);
                }

                std::vector<int> assign_blk;
                std::vector<float> dot_unused;
                for (int i0 = 0; i0 < n; i0 += block_cols) {
                    const int cols = std::min(block_cols, n - i0);
                    DeviceMatF32View Xd;
                    if (have_full_xnorm && Xnorm_full.ptr && Xnorm_full.rows == d && Xnorm_full.cols == n) {
                        Xd.ptr = Xnorm_full.ptr +
                            static_cast<std::size_t>(i0) * static_cast<std::size_t>(Xnorm_full.ld);
                        Xd.rows = d;
                        Xd.cols = cols;
                        Xd.ld = Xnorm_full.ld;
                    }
                    else if (overlap_h2d_pass2) {
                        cuda_kernels->KmeansComputeWaitForUpload(overlap_slot_cur_pass2);
                        Xd = overlap_Xd_pass2[overlap_slot_cur_pass2];
                    }
                    else {
                        const float* xptr_f32 = (X_pinned_base ? X_pinned_base : X.data.data()) +
                            static_cast<std::size_t>(i0) * static_cast<std::size_t>(d);
                        const std::uint8_t* xptr_u8 = X_u8_base
                                                          ? (X_u8_base + static_cast<std::size_t>(i0) *
                                                              static_cast<std::size_t>(d))
                                                          : nullptr;
                        if (use_u8_upload) {
                            Xd = cuda_kernels->KmeansUploadU8AndNormalizeHostPtrBAsync(
                                /*slot=*/0,
                                         xptr_u8, /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols,
                                         /*stage_to_pinned=*/cfg.pin_host_x,
                                         /*stage_s=*/nullptr,
                                         /*h2d_s=*/nullptr,
                                         /*kernel_s=*/nullptr);
                            cuda_kernels->KmeansComputeWaitForUpload(0);
                        }
                        else {
                            Xd = cuda_kernels->KmeansUploadAndNormalizeHostPtrBTimed(
                                xptr_f32, /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols,
                                /*stage_to_pinned=*/(X_pinned_base == nullptr),
                                /*stage_s=*/nullptr,
                                /*h2d_s=*/nullptr,
                                /*kernel_s=*/nullptr);
                        }
                    }

                    if (use_hier2) {
                        cuda_kernels->KmeansHier2AssignColsF32Device(Xd, coarse, out.centers,
                                                                     /*K_valid=*/k,
                                                                     /*K1=*/hier2.K1,
                                                                     /*K2=*/hier2.K2,
                                                                     /*topL=*/cfg.large_k_top_coarse);
                    }
                    else {
                        constexpr std::size_t kScoreBudgetBytes = 256ull * 1024ull * 1024ull;
                        int tile_n = static_cast<int>(kScoreBudgetBytes /
                            (static_cast<std::size_t>(k) * sizeof(float)));
                        tile_n = std::min(tile_n, cols);
                        if (tile_n > 256) tile_n = (tile_n / 256) * 256;
                        tile_n = std::max(1, tile_n);

                        cuda_kernels->KmeansEnsureBlockArgmax(cols);
                        DeviceVecI32View d_assign_tile;
                        DeviceVecF32View d_best_tile;
                        for (int j0 = 0; j0 < cols; j0 += tile_n) {
                            const int tcols = std::min(tile_n, cols - j0);
                            const float* dX =
                                Xd.ptr + static_cast<std::size_t>(j0) * static_cast<std::size_t>(Xd.ld);
                            DeviceMatF32View scores;
                            stream_kernels->GemmDeviceDevicePtrB(/*transA=*/true, /*transB=*/false,
                                                                            /*alpha=*/1.0f,
                                                                            out.centers,
                                                                            dX,
                                                                            /*ldB=*/Xd.ld,
                                                                            /*rowsB=*/Xd.rows,
                                                                            /*colsB=*/tcols,
                                                                            /*beta=*/0.0f,
                                                                            &scores);
                            stream_kernels->ArgmaxColsF32Device(scores, &d_assign_tile, &d_best_tile);
                            cuda_kernels->KmeansCopyTileArgmaxToBlock(d_assign_tile, d_best_tile, j0);
                        }
                    }
                    if (overlap_h2d_pass2) {
                        const int i0_next = i0 + block_cols;
                        if (i0_next < n) {
                            const int cols_next = std::min(block_cols, n - i0_next);
                            overlap_Xd_pass2[overlap_slot_next_pass2] =
                                ScheduleUploadPass2(overlap_slot_next_pass2, i0_next, cols_next);
                        }
                    }

                    if (cfg.collect_metrics) {
                        cuda_kernels->KmeansAccumulateMseFromBlockBestDot(cols);
                    }
                    if (onepass_active) {
                        cuda_kernels->KmeansAccumulateCostHistogramFromBestDot(
                            cuda_kernels->KmeansBlockBestView(), bins);
                    }
                    cuda_kernels->KmeansAnnealUpdateWeightsAndTotalsFromBlockBest(
                        /*col0=*/i0, cols,
                                 /*threshold=*/effective_threshold,
                                 /*annealed_factor=*/annealed_factor,
                                 /*min_weight=*/cfg.min_weight);
                    const DeviceVecF32View d_w = cuda_kernels->KmeansAnnealWeightsBlockView(i0, cols);
                    cuda_kernels->KmeansAccumulateWeightsDeviceAssignNoAtomic(
                        Xd, cuda_kernels->KmeansBlockAssignView(), d_w);

                    // Host assignments are required by the API contract; download for this block.
                    cuda_kernels->KmeansDownloadBlockArgmax(&assign_blk, &dot_unused);
                    for (int j = 0; j < cols; ++j) {
                        out.assignments[static_cast<std::size_t>(i0 + j)] = assign_blk[static_cast<std::size_t>(j)];
                    }

                    if (use_u8_upload && !overlap_h2d_pass2) {
                        cuda_kernels->KmeansMarkUploadSlotConsumed(0);
                    }
                    if (overlap_h2d_pass2) {
                        cuda_kernels->KmeansMarkUploadSlotConsumed(overlap_slot_cur_pass2);
                        std::swap(overlap_slot_cur_pass2, overlap_slot_next_pass2);
                    }
                }

                double total_cost = 0.0;
                double total_w = 0.0;
                cuda_kernels->KmeansDownloadAnnealTotalsAsync();
                if (cfg.collect_metrics) {
                    cuda_kernels->KmeansDownloadMetricTotalsAsync();
                }
                cuda_kernels->KmeansDownloadSums(&sum_x, &sum_w);
                cuda_kernels->Sync();
                cuda_kernels->KmeansSyncAnnealTotalsDownload(&total_w, &total_cost);
                if (timing) {
                    if (!cfg.collect_metrics) {
                        timing->mse_proxy = -1.0;
                    }
                    else {
                        double mse_sum = 0.0;
                        double mse_count = 0.0;
                        cuda_kernels->KmeansSyncMetricTotalsDownload(&mse_sum, &mse_count);
                        timing->mse_proxy = (mse_count > 0.0) ? (mse_sum / mse_count) : -1.0;
                    }
                }

                if (onepass_active) {
                    std::vector<std::uint32_t> hist_u32(static_cast<std::size_t>(bins));
                    cuda_kernels->KmeansDownloadCostHistogramAsync(hist_u32.data(), bins);
                    cuda_kernels->KmeansDownloadCostHistogramSync();
                    const std::uint64_t target = static_cast<std::uint64_t>(
                        cfg.outlier_quantile * static_cast<float>(std::max(1, n - 1)));
                    std::uint64_t prefix = 0;
                    int bq = bins - 1;
                    for (int b = 0; b < bins; ++b) {
                        prefix += static_cast<std::uint64_t>(hist_u32[static_cast<std::size_t>(b)]);
                        if (prefix > target) {
                            bq = b;
                            break;
                        }
                    }
                    const float bw = (hi - lo) / static_cast<float>(bins);
                    const float quantile_threshold = lo + (static_cast<float>(bq) + 0.5f) * bw;
                    threshold_prev = std::max(quantile_threshold, cfg.cost_threshold);
                }

                const float current_cost = (total_w > 0.0) ? static_cast<float>(total_cost / total_w) : 0.0f;

                // Finalize centers from sums.
#pragma omp parallel for default(none) shared(out, sum_w, fallback, X, sum_x) firstprivate(d, k, kEps) schedule(static)
                for (int c = 0; c < k; ++c) {
                    float* center = out.centers.Col(c);
                    const float w = sum_w[static_cast<std::size_t>(c)];
                    if (w <= kEps) {
                        const int idx = fallback[static_cast<std::size_t>(c)];
                        const float* xi = X.Col(idx);
                        float norm = 0.0f;
#pragma omp simd reduction(+:norm)
                        for (int r = 0; r < d; ++r) norm += xi[r] * xi[r];
                        norm = std::sqrt(std::max(norm, kEps));
                        const float inv = (norm > kEps) ? (1.0f / norm) : 1.0f;
#pragma omp simd
                        for (int r = 0; r < d; ++r) center[r] = xi[r] * inv;
                        continue;
                    }
                    const float inv_w = 1.0f / w;
                    const float* sx = sum_x.Col(c);
                    float norm = 0.0f;
                    for (int r = 0; r < d; ++r) {
                        const float v = sx[r] * inv_w;
                        center[r] = v;
                        norm += v * v;
                    }
                    norm = std::sqrt(std::max(norm, kEps));
                    const float inv = 1.0f / norm;
#pragma omp simd
                    for (int r = 0; r < d; ++r) center[r] *= inv;
                }

                if (std::abs(prev_cost - current_cost) < cfg.tol) {
                    break;
                }
                prev_cost = current_cost;
                continue;
            }
            if (cfg.anneal_no_spill) {
                const std::string tmp_dir = !cfg.tmp_dir.empty() ? cfg.tmp_dir : std::string("./tmp");
                std::filesystem::create_directories(tmp_dir);
                const std::string w_in_path = tmp_dir + "/kmeans_weights_in_f32.bin";
                const std::string w_out_path = tmp_dir + "/kmeans_weights_out_f32.bin";

                const int bins = std::max(1, cfg.quantile_bins);
                const float lo = 0.0f;
                const float hi = 2.0f;
                const float inv_span = static_cast<float>(bins) / (hi - lo);

                // Pass 1: scan all samples and build a cost histogram to estimate the outlier quantile threshold.
                std::vector<std::uint32_t> hist_u32;
                std::vector<std::uint64_t> hist_u64;
                if (cuda_kernels) {
                    cuda_kernels->KmeansResetCostHistogram(bins);
                }
                else {
                    hist_u64.assign(static_cast<std::size_t>(bins), 0);
                }

                const bool overlap_h2d_pass1 =
                    (cuda_kernels != nullptr) && (!have_full_xnorm) && (block_cols >= 4096);
                if (timing && overlap_h2d_pass1) {
                    timing->used_overlap_pass1 = 1;
                }
                int overlap_slot_cur_pass1 = 0;
                int overlap_slot_next_pass1 = 1;
                DeviceMatF32View overlap_Xd_pass1[2];
                auto ScheduleUploadPass1 = [&](int slot, int i0, int cols_upload) -> DeviceMatF32View
                {
                    if (use_u8_upload) {
                        const std::uint8_t* xptr =
                            X_u8_base + static_cast<std::size_t>(i0) * static_cast<std::size_t>(d);
                        return cuda_kernels->KmeansUploadU8AndNormalizeHostPtrBAsync(
                            slot,
                            xptr, /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols_upload,
                            /*stage_to_pinned=*/cfg.pin_host_x,
                            /*stage_s=*/nullptr,
                            /*h2d_s=*/nullptr,
                            /*kernel_s=*/nullptr);
                    }
                    else {
                        const float* xptr =
                            (X_pinned_base ? X_pinned_base : X.data.data()) +
                            static_cast<std::size_t>(i0) * static_cast<std::size_t>(d);
                        if (X_pinned_base) {
                            return cuda_kernels->KmeansUploadAndNormalizeHostPtrBAsync(
                                slot,
                                xptr, /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols_upload,
                                /*stage_to_pinned=*/false,
                                /*stage_s=*/nullptr,
                                /*h2d_s=*/nullptr,
                                /*kernel_s=*/nullptr);
                        }
                        float* pinned = cuda_kernels->KmeansGetPinnedStageBuffer(slot, d, cols_upload);
                        std::memcpy(pinned,
                                    xptr,
                                    sizeof(float) * static_cast<std::size_t>(d) * static_cast<std::size_t>(
                                        cols_upload));
                        return cuda_kernels->KmeansUploadAndNormalizePinnedAsync(
                            slot,
                            pinned, /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols_upload,
                            /*h2d_s=*/nullptr,
                            /*kernel_s=*/nullptr);
                    }
                };
                if (overlap_h2d_pass1 && n > 0) {
                    const int cols0 = std::min(block_cols, n);
                    overlap_slot_cur_pass1 = 0;
                    overlap_slot_next_pass1 = 1;
                    overlap_Xd_pass1[overlap_slot_cur_pass1] = ScheduleUploadPass1(overlap_slot_cur_pass1, 0, cols0);
                }

                std::vector<int> assign_blk;
                std::vector<float> dot_blk;
                for (int i0 = 0; i0 < n; i0 += block_cols) {
                    const int cols = std::min(block_cols, n - i0);
                    if (cuda_kernels) {
                        DeviceMatF32View Xd;
                        if (have_full_xnorm && Xnorm_full.ptr && Xnorm_full.rows == d && Xnorm_full.cols == n) {
                            Xd.ptr = Xnorm_full.ptr +
                                static_cast<std::size_t>(i0) * static_cast<std::size_t>(Xnorm_full.ld);
                            Xd.rows = d;
                            Xd.cols = cols;
                            Xd.ld = Xnorm_full.ld;
                        }
                        else {
                            if (overlap_h2d_pass1) {
                                cuda_kernels->KmeansComputeWaitForUpload(overlap_slot_cur_pass1);
                                Xd = overlap_Xd_pass1[overlap_slot_cur_pass1];
                            }
                            else {
                                Timer t_norm;
                                const float* xptr_f32 = (X_pinned_base ? X_pinned_base : X.data.data()) +
                                    static_cast<std::size_t>(i0) * static_cast<std::size_t>(d);
                                const std::uint8_t* xptr_u8 = X_u8_base
                                                                  ? (X_u8_base + static_cast<std::size_t>(i0) *
                                                                      static_cast<std::size_t>(d))
                                                                  : nullptr;
                                double stage_s = 0.0;
                                double h2d_s = 0.0;
                                double ker_s = 0.0;
                                if (profile_timing && timing) {
                                    if (use_u8_upload) {
                                        Xd = cuda_kernels->KmeansUploadU8AndNormalizeHostPtrBAsync(
                                            /*slot=*/0,
                                                     xptr_u8, /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols,
                                                     /*stage_to_pinned=*/cfg.pin_host_x,
                                                     /*stage_s=*/&stage_s, &h2d_s, &ker_s);
                                        cuda_kernels->KmeansComputeWaitForUpload(0);
                                    }
                                    else {
                                        Xd = cuda_kernels->KmeansUploadAndNormalizeHostPtrBTimed(
                                            xptr_f32, /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols,
                                            /*stage_to_pinned=*/(X_pinned_base == nullptr),
                                            /*stage_s=*/&stage_s, &h2d_s, &ker_s);
                                    }
                                }
                                else {
                                    if (use_u8_upload) {
                                        Xd = cuda_kernels->KmeansUploadU8AndNormalizeHostPtrBAsync(
                                            /*slot=*/0,
                                                     xptr_u8, /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols,
                                                     /*stage_to_pinned=*/cfg.pin_host_x,
                                                     /*stage_s=*/nullptr,
                                                     /*h2d_s=*/nullptr,
                                                     /*kernel_s=*/nullptr);
                                        cuda_kernels->KmeansComputeWaitForUpload(0);
                                    }
                                    else {
                                        Xd = cuda_kernels->KmeansUploadAndNormalizeHostPtrBTimed(
                                            xptr_f32, /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols,
                                            /*stage_to_pinned=*/(X_pinned_base == nullptr),
                                            /*stage_s=*/nullptr,
                                            /*h2d_s=*/nullptr,
                                            /*kernel_s=*/nullptr);
                                    }
                                }
                                if (profile_timing && timing) {
                                    timing->normalize_stage_s += stage_s;
                                    timing->normalize_h2d_s += h2d_s;
                                    timing->normalize_kernel_s += ker_s;
                                    timing->normalize_s += t_norm.ElapsedSeconds();
                                }
                            }
                        }

                        constexpr std::size_t kScoreBudgetBytes = 256ull * 1024ull * 1024ull; // 256 MiB
                        int tile_n = static_cast<int>(kScoreBudgetBytes /
                            (sizeof(float) * static_cast<std::size_t>(std::max(1, k))));
                        tile_n = std::max(1, tile_n);
                        tile_n = std::min(tile_n, cols);
                        if (tile_n > 256) tile_n = (tile_n / 256) * 256;
                        tile_n = std::max(1, tile_n);

                        cuda_kernels->KmeansEnsureBlockArgmax(cols);
                        DeviceVecI32View d_assign_tile;
                        DeviceVecF32View d_best_tile;
                        for (int j0 = 0; j0 < cols; j0 += tile_n) {
                            const int tcols = std::min(tile_n, cols - j0);
                            const float* dX = Xd.ptr + static_cast<std::size_t>(j0) * static_cast<std::size_t>(Xd.ld);

                            DeviceMatF32View scores;
                            Timer t_gemm;
                            stream_kernels->GemmDeviceDevicePtrB(/*transA=*/true, /*transB=*/false,
                                                                            /*alpha=*/1.0f,
                                                                            out.centers,
                                                                            dX,
                                                                            /*ldB=*/Xd.ld,
                                                                            /*rowsB=*/Xd.rows,
                                                                            /*colsB=*/tcols,
                                                                            /*beta=*/0.0f,
                                                                            &scores);
                            if (profile_timing && timing) {
                                stream_kernels->SyncCompute();
                                timing->assign_gemm_s += t_gemm.ElapsedSeconds();
                            }

                            Timer t_arg;
                            stream_kernels->ArgmaxColsF32Device(scores, &d_assign_tile, &d_best_tile);
                            cuda_kernels->KmeansCopyTileArgmaxToBlock(d_assign_tile, d_best_tile, j0);
                            if (profile_timing && timing) {
                                stream_kernels->SyncCompute();
                                timing->assign_argmax_s += t_arg.ElapsedSeconds();
                            }
                        }
                        if (overlap_h2d_pass1) {
                            const int i0_next = i0 + block_cols;
                            if (i0_next < n) {
                                const int cols_next = std::min(block_cols, n - i0_next);
                                overlap_Xd_pass1[overlap_slot_next_pass1] =
                                    ScheduleUploadPass1(overlap_slot_next_pass1, i0_next, cols_next);
                            }
                        }
                        cuda_kernels->KmeansAccumulateCostHistogramFromBestDot(
                            cuda_kernels->KmeansBlockBestView(), bins);
                        if (profile_timing && timing) {
                            stream_kernels->SyncCompute();
                        }
                        if (use_u8_upload && !overlap_h2d_pass1) {
                            cuda_kernels->KmeansMarkUploadSlotConsumed(0);
                        }
                    }
                    else {
                        if (Xblk_norm.cols != cols) {
                            Xblk_norm = ColMajorMatrix<float>(d, cols);
                        }
                        if (profile_timing && timing) {
                            Timer t_norm;
#pragma omp parallel for default(none) shared(X, Xblk_norm) firstprivate(i0, cols, d, kEps) schedule(static)
                            for (int j = 0; j < cols; ++j) {
                                const float* xi = X.Col(i0 + j);
                                float* dst = Xblk_norm.Col(j);
                                float norm = 0.0f;
#pragma omp simd reduction(+:norm)
                                for (int r = 0; r < d; ++r) norm += xi[r] * xi[r];
                                norm = std::sqrt(std::max(norm, kEps));
                                const float inv = (norm > kEps) ? (1.0f / norm) : 1.0f;
#pragma omp simd
                                for (int r = 0; r < d; ++r) dst[r] = xi[r] * inv;
                            }
                            timing->normalize_s += t_norm.ElapsedSeconds();
                        }
                        else {
#pragma omp parallel for default(none) shared(X, Xblk_norm) firstprivate(i0, cols, d, kEps) schedule(static)
                            for (int j = 0; j < cols; ++j) {
                                const float* xi = X.Col(i0 + j);
                                float* dst = Xblk_norm.Col(j);
                                float norm = 0.0f;
#pragma omp simd reduction(+:norm)
                                for (int r = 0; r < d; ++r) norm += xi[r] * xi[r];
                                norm = std::sqrt(std::max(norm, kEps));
                                const float inv = (norm > kEps) ? (1.0f / norm) : 1.0f;
#pragma omp simd
                                for (int r = 0; r < d; ++r) dst[r] = xi[r] * inv;
                            }
                        }
                        AssignSamplesGpu(Xblk_norm, out.centers, stream_kernels,
                                         profile_timing, timing,
                                         &assign_blk, &dot_blk);
                        for (int j = 0; j < cols; ++j) {
                            float x = 1.0f - dot_blk[static_cast<std::size_t>(j)];
                            if (x < lo) x = lo;
                            if (x > hi) x = hi;
                            int b = static_cast<int>((x - lo) * inv_span);
                            if (b >= bins) b = bins - 1;
                            hist_u64[static_cast<std::size_t>(b)] += 1;
                        }
                    }
                    if (overlap_h2d_pass1) {
                        cuda_kernels->KmeansMarkUploadSlotConsumed(overlap_slot_cur_pass1);
                        std::swap(overlap_slot_cur_pass1, overlap_slot_next_pass1);
                    }
                }

                if (cuda_kernels) {
                    if (profile_timing && timing) {
                        Timer t_hist;
                        hist_u32.resize(static_cast<std::size_t>(bins));
                        cuda_kernels->KmeansDownloadCostHistogramAsync(hist_u32.data(), bins);
                        cuda_kernels->KmeansDownloadCostHistogramSync();
                        timing->weights_s += t_hist.ElapsedSeconds();
                    }
                    else {
                        hist_u32.resize(static_cast<std::size_t>(bins));
                        cuda_kernels->KmeansDownloadCostHistogramAsync(hist_u32.data(), bins);
                        cuda_kernels->KmeansDownloadCostHistogramSync();
                    }
                }

                const std::uint64_t target = static_cast<std::uint64_t>(
                    cfg.outlier_quantile * static_cast<float>(std::max(1, n - 1)));
                std::uint64_t prefix = 0;
                int bq = bins - 1;
                for (int b = 0; b < bins; ++b) {
                    if (cuda_kernels) {
                        prefix += static_cast<std::uint64_t>(hist_u32[static_cast<std::size_t>(b)]);
                    }
                    else {
                        prefix += hist_u64[static_cast<std::size_t>(b)];
                    }
                    if (prefix > target) {
                        bq = b;
                        break;
                    }
                }
                const float bw = (hi - lo) / static_cast<float>(bins);
                const float quantile_threshold = lo + (static_cast<float>(bq) + 0.5f) * bw;
                const float effective_threshold = std::max(quantile_threshold, cfg.cost_threshold);

                // Pass 2: rescan, recompute argmax, update weights, and accumulate centers.
                const float denom = static_cast<float>(std::max(1, cfg.max_iters - cfg.warmup_iters));
                const float annealed_factor =
                    cfg.annealing_factor * (1.0f - std::exp(-(iter - cfg.warmup_iters) / denom));

                float current_cost = 0.0f;
                std::vector<float> w_blk;
                double total_cost = 0.0;
                double total_w = 0.0;

                std::ifstream f_w_in(w_in_path, std::ios::binary);
                if (!f_w_in) throw std::runtime_error("Failed to open: " + w_in_path);
                std::ofstream f_w_out(w_out_path, std::ios::binary | std::ios::trunc);
                if (!f_w_out) throw std::runtime_error("Failed to open: " + w_out_path);

                ColMajorMatrix<float> sum_x(d, k);
                std::vector<float> sum_w(static_cast<std::size_t>(k), 0.0f);
                if (cuda_kernels) {
                    cuda_kernels->KmeansResetSums(d, k);
                }
                else {
                    sum_x.data.assign(static_cast<std::size_t>(d) * static_cast<std::size_t>(k), 0.0f);
                }

                std::vector<int> offsets(static_cast<std::size_t>(k) + 1, 0);
                std::vector<int> cursor(static_cast<std::size_t>(k) + 1, 0);
                std::vector<int> order;

                const bool overlap_h2d_pass2 =
                    (cuda_kernels != nullptr) && (!have_full_xnorm) && (block_cols >= 4096);
                if (timing && overlap_h2d_pass2) {
                    timing->used_overlap_pass2 = 1;
                }
                int overlap_slot_cur_pass2 = 0;
                int overlap_slot_next_pass2 = 1;
                DeviceMatF32View overlap_Xd_pass2[2];
                auto ScheduleUploadPass2 = [&](int slot, int i0, int cols_upload) -> DeviceMatF32View
                {
                    if (use_u8_upload) {
                        const std::uint8_t* xptr =
                            X_u8_base + static_cast<std::size_t>(i0) * static_cast<std::size_t>(d);
                        return cuda_kernels->KmeansUploadU8AndNormalizeHostPtrBAsync(
                            slot,
                            xptr, /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols_upload,
                            /*stage_to_pinned=*/cfg.pin_host_x,
                            /*stage_s=*/nullptr,
                            /*h2d_s=*/nullptr,
                            /*kernel_s=*/nullptr);
                    }
                    else {
                        const float* xptr =
                            (X_pinned_base ? X_pinned_base : X.data.data()) +
                            static_cast<std::size_t>(i0) * static_cast<std::size_t>(d);
                        if (X_pinned_base) {
                            return cuda_kernels->KmeansUploadAndNormalizeHostPtrBAsync(
                                slot,
                                xptr, /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols_upload,
                                /*stage_to_pinned=*/false,
                                /*stage_s=*/nullptr,
                                /*h2d_s=*/nullptr,
                                /*kernel_s=*/nullptr);
                        }
                        float* pinned = cuda_kernels->KmeansGetPinnedStageBuffer(slot, d, cols_upload);
                        std::memcpy(pinned,
                                    xptr,
                                    sizeof(float) * static_cast<std::size_t>(d) * static_cast<std::size_t>(
                                        cols_upload));
                        return cuda_kernels->KmeansUploadAndNormalizePinnedAsync(
                            slot,
                            pinned, /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols_upload,
                            /*h2d_s=*/nullptr,
                            /*kernel_s=*/nullptr);
                    }
                };
                if (overlap_h2d_pass2 && n > 0) {
                    const int cols0 = std::min(block_cols, n);
                    overlap_slot_cur_pass2 = 0;
                    overlap_slot_next_pass2 = 1;
                    overlap_Xd_pass2[overlap_slot_cur_pass2] = ScheduleUploadPass2(overlap_slot_cur_pass2, 0, cols0);
                }

                double acc2_s = 0.0;
                for (int i0 = 0; i0 < n; i0 += block_cols) {
                    const int cols = std::min(block_cols, n - i0);
                    if (profile_timing && timing) {
                        Timer t_wread;
                        ReadVectorExactOrThrow(f_w_in, &w_blk, static_cast<std::size_t>(cols));
                        timing->update_read_weights_s += t_wread.ElapsedSeconds();
                    }
                    else {
                        ReadVectorExactOrThrow(f_w_in, &w_blk, static_cast<std::size_t>(cols));
                    }

                    if (cuda_kernels) {
                        DeviceMatF32View Xd;
                        if (have_full_xnorm && Xnorm_full.ptr && Xnorm_full.rows == d && Xnorm_full.cols == n) {
                            Xd.ptr = Xnorm_full.ptr +
                                static_cast<std::size_t>(i0) * static_cast<std::size_t>(Xnorm_full.ld);
                            Xd.rows = d;
                            Xd.cols = cols;
                            Xd.ld = Xnorm_full.ld;
                        }
                        else {
                            if (overlap_h2d_pass2) {
                                cuda_kernels->KmeansComputeWaitForUpload(overlap_slot_cur_pass2);
                                Xd = overlap_Xd_pass2[overlap_slot_cur_pass2];
                            }
                            else {
                                Timer t_norm2;
                                double stage_s = 0.0;
                                double h2d_s = 0.0;
                                double ker_s = 0.0;
                                if (use_u8_upload) {
                                    const std::uint8_t* xptr =
                                        X_u8_base + static_cast<std::size_t>(i0) * static_cast<std::size_t>(d);
                                    Xd = cuda_kernels->KmeansUploadU8AndNormalizeHostPtrBAsync(
                                        /*slot=*/0,
                                                 xptr, /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols,
                                                 /*stage_to_pinned=*/cfg.pin_host_x,
                                                 /*stage_s=*/(profile_timing && timing) ? &stage_s : nullptr,
                                                 /*h2d_s=*/(profile_timing && timing) ? &h2d_s : nullptr,
                                                 /*kernel_s=*/(profile_timing && timing) ? &ker_s : nullptr);
                                    cuda_kernels->KmeansComputeWaitForUpload(0);
                                }
                                else {
                                    const float* xptr = (X_pinned_base ? X_pinned_base : X.data.data()) +
                                        static_cast<std::size_t>(i0) * static_cast<std::size_t>(d);
                                    if (profile_timing && timing) {
                                        Xd = cuda_kernels->KmeansUploadAndNormalizeHostPtrBTimed(
                                            xptr, /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols,
                                            /*stage_to_pinned=*/(X_pinned_base == nullptr),
                                            /*stage_s=*/&stage_s, &h2d_s, &ker_s);
                                    }
                                    else {
                                        Xd = cuda_kernels->KmeansUploadAndNormalizeHostPtrBTimed(
                                            xptr, /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols,
                                            /*stage_to_pinned=*/(X_pinned_base == nullptr),
                                            /*stage_s=*/nullptr,
                                            /*h2d_s=*/nullptr,
                                            /*kernel_s=*/nullptr);
                                    }
                                }
                                if (profile_timing && timing) {
                                    timing->normalize_stage_s += stage_s;
                                    timing->normalize_h2d_s += h2d_s;
                                    timing->normalize_kernel_s += ker_s;
                                    timing->normalize_s += t_norm2.ElapsedSeconds();
                                }
                            }
                        }

                        constexpr std::size_t kScoreBudgetBytes = 256ull * 1024ull * 1024ull; // 256 MiB
                        int tile_n = static_cast<int>(kScoreBudgetBytes /
                            (sizeof(float) * static_cast<std::size_t>(std::max(1, k))));
                        tile_n = std::max(1, tile_n);
                        tile_n = std::min(tile_n, cols);
                        if (tile_n > 256) tile_n = (tile_n / 256) * 256;
                        tile_n = std::max(1, tile_n);

                        cuda_kernels->KmeansEnsureBlockArgmax(cols);
                        DeviceVecI32View d_assign_tile;
                        DeviceVecF32View d_best_tile;
                        for (int j0 = 0; j0 < cols; j0 += tile_n) {
                            const int tcols = std::min(tile_n, cols - j0);
                            const float* dX = Xd.ptr + static_cast<std::size_t>(j0) * static_cast<std::size_t>(Xd.ld);

                            DeviceMatF32View scores;
                            Timer t_gemm;
                            stream_kernels->GemmDeviceDevicePtrB(/*transA=*/true, /*transB=*/false,
                                                                            /*alpha=*/1.0f,
                                                                            out.centers,
                                                                            dX,
                                                                            /*ldB=*/Xd.ld,
                                                                            /*rowsB=*/Xd.rows,
                                                                            /*colsB=*/tcols,
                                                                            /*beta=*/0.0f,
                                                                            &scores);
                            if (profile_timing && timing) {
                                stream_kernels->SyncCompute();
                                timing->assign_gemm_s += t_gemm.ElapsedSeconds();
                            }

                            Timer t_arg;
                            stream_kernels->ArgmaxColsF32Device(scores, &d_assign_tile, &d_best_tile);
                            cuda_kernels->KmeansCopyTileArgmaxToBlock(d_assign_tile, d_best_tile, j0);
                            if (profile_timing && timing) {
                                stream_kernels->SyncCompute();
                                timing->assign_argmax_s += t_arg.ElapsedSeconds();
                            }
                        }
                        if (overlap_h2d_pass2) {
                            const int i0_next = i0 + block_cols;
                            if (i0_next < n) {
                                const int cols_next = std::min(block_cols, n - i0_next);
                                overlap_Xd_pass2[overlap_slot_next_pass2] =
                                    ScheduleUploadPass2(overlap_slot_next_pass2, i0_next, cols_next);
                            }
                        }
                        cuda_kernels->KmeansDownloadBlockArgmax(&assign_blk, &dot_blk);
                        if (use_u8_upload && !overlap_h2d_pass2) {
                            cuda_kernels->KmeansMarkUploadSlotConsumed(0);
                        }

                        if (profile_timing && timing) {
                            Timer t_wupd;
                            for (int j = 0; j < cols; ++j) {
                                const float cost = 1.0f - dot_blk[static_cast<std::size_t>(j)];
                                float w = w_blk[static_cast<std::size_t>(j)];
                                if (cost > effective_threshold) {
                                    w = std::max(cfg.min_weight, w * (1.0f - annealed_factor));
                                }
                                else {
                                    w = std::min(1.0f, w * (1.0f + annealed_factor * 0.5f));
                                }
                                w_blk[static_cast<std::size_t>(j)] = w;
                                out.assignments[static_cast<std::size_t>(i0 + j)] =
                                    assign_blk[static_cast<std::size_t>(j)];
                                total_cost += static_cast<double>(w) * static_cast<double>(cost);
                                total_w += static_cast<double>(w);
                            }
                            f_w_out.write(reinterpret_cast<const char*>(w_blk.data()),
                                          static_cast<std::streamsize>(sizeof(float) * w_blk.size()));
                            timing->weights_s += t_wupd.ElapsedSeconds();
                        }
                        else {
                            for (int j = 0; j < cols; ++j) {
                                const float cost = 1.0f - dot_blk[static_cast<std::size_t>(j)];
                                float w = w_blk[static_cast<std::size_t>(j)];
                                if (cost > effective_threshold) {
                                    w = std::max(cfg.min_weight, w * (1.0f - annealed_factor));
                                }
                                else {
                                    w = std::min(1.0f, w * (1.0f + annealed_factor * 0.5f));
                                }
                                w_blk[static_cast<std::size_t>(j)] = w;
                                out.assignments[static_cast<std::size_t>(i0 + j)] =
                                    assign_blk[static_cast<std::size_t>(j)];
                                total_cost += static_cast<double>(w) * static_cast<double>(cost);
                                total_w += static_cast<double>(w);
                            }
                            f_w_out.write(reinterpret_cast<const char*>(w_blk.data()),
                                          static_cast<std::streamsize>(sizeof(float) * w_blk.size()));
                        }

                        Timer t_acc;
                        const DeviceVecF32View d_w = cuda_kernels->KmeansUploadBlockWeights(w_blk);
                        cuda_kernels->KmeansAccumulateWeightsDeviceAssignNoAtomic(
                            Xd, cuda_kernels->KmeansBlockAssignView(), d_w);
                        if (profile_timing && timing) {
                            cuda_kernels->SyncCompute();
                        }
                        acc2_s += t_acc.ElapsedSeconds();
                    }
                    else {
                        if (Xblk_norm.cols != cols) {
                            Xblk_norm = ColMajorMatrix<float>(d, cols);
                        }
                        if (profile_timing && timing) {
                            Timer t_norm2;
#pragma omp parallel for default(none) shared(X, Xblk_norm) firstprivate(i0, cols, d, kEps) schedule(static)
                            for (int j = 0; j < cols; ++j) {
                                const float* xi = X.Col(i0 + j);
                                float* dst = Xblk_norm.Col(j);
                                float norm = 0.0f;
#pragma omp simd reduction(+:norm)
                                for (int r = 0; r < d; ++r) norm += xi[r] * xi[r];
                                norm = std::sqrt(std::max(norm, kEps));
                                const float inv = (norm > kEps) ? (1.0f / norm) : 1.0f;
#pragma omp simd
                                for (int r = 0; r < d; ++r) dst[r] = xi[r] * inv;
                            }
                            timing->normalize_s += t_norm2.ElapsedSeconds();
                        }
                        else {
#pragma omp parallel for default(none) shared(X, Xblk_norm) firstprivate(i0, cols, d, kEps) schedule(static)
                            for (int j = 0; j < cols; ++j) {
                                const float* xi = X.Col(i0 + j);
                                float* dst = Xblk_norm.Col(j);
                                float norm = 0.0f;
#pragma omp simd reduction(+:norm)
                                for (int r = 0; r < d; ++r) norm += xi[r] * xi[r];
                                norm = std::sqrt(std::max(norm, kEps));
                                const float inv = (norm > kEps) ? (1.0f / norm) : 1.0f;
#pragma omp simd
                                for (int r = 0; r < d; ++r) dst[r] = xi[r] * inv;
                            }
                        }

                        AssignSamplesGpu(Xblk_norm, out.centers, stream_kernels,
                                         profile_timing, timing,
                                         &assign_blk, &dot_blk);

                        if (profile_timing && timing) {
                            Timer t_wupd;
                            for (int j = 0; j < cols; ++j) {
                                const float cost = 1.0f - dot_blk[static_cast<std::size_t>(j)];
                                float w = w_blk[static_cast<std::size_t>(j)];
                                if (cost > effective_threshold) {
                                    w = std::max(cfg.min_weight, w * (1.0f - annealed_factor));
                                }
                                else {
                                    w = std::min(1.0f, w * (1.0f + annealed_factor * 0.5f));
                                }
                                w_blk[static_cast<std::size_t>(j)] = w;
                                out.assignments[static_cast<std::size_t>(i0 + j)] =
                                    assign_blk[static_cast<std::size_t>(j)];
                                total_cost += static_cast<double>(w) * static_cast<double>(cost);
                                total_w += static_cast<double>(w);
                            }
                            f_w_out.write(reinterpret_cast<const char*>(w_blk.data()),
                                          static_cast<std::streamsize>(sizeof(float) * w_blk.size()));
                            timing->weights_s += t_wupd.ElapsedSeconds();
                        }
                        else {
                            for (int j = 0; j < cols; ++j) {
                                const float cost = 1.0f - dot_blk[static_cast<std::size_t>(j)];
                                float w = w_blk[static_cast<std::size_t>(j)];
                                if (cost > effective_threshold) {
                                    w = std::max(cfg.min_weight, w * (1.0f - annealed_factor));
                                }
                                else {
                                    w = std::min(1.0f, w * (1.0f + annealed_factor * 0.5f));
                                }
                                w_blk[static_cast<std::size_t>(j)] = w;
                                out.assignments[static_cast<std::size_t>(i0 + j)] =
                                    assign_blk[static_cast<std::size_t>(j)];
                                total_cost += static_cast<double>(w) * static_cast<double>(cost);
                                total_w += static_cast<double>(w);
                            }
                            f_w_out.write(reinterpret_cast<const char*>(w_blk.data()),
                                          static_cast<std::streamsize>(sizeof(float) * w_blk.size()));
                        }

                        Timer t_acc;
                        std::fill(offsets.begin(), offsets.end(), 0);
                        for (int j = 0; j < cols; ++j) {
                            ++offsets[static_cast<std::size_t>(assign_blk[static_cast<std::size_t>(j)]) + 1];
                        }
                        for (int c = 0; c < k; ++c)
                            offsets[static_cast<std::size_t>(c) + 1] += offsets[static_cast<
                                std::size_t>(c)];
                        cursor = offsets;
                        order.resize(static_cast<std::size_t>(cols));
                        for (int j = 0; j < cols; ++j) {
                            const int c = assign_blk[static_cast<std::size_t>(j)];
                            const int pos = cursor[static_cast<std::size_t>(c)]++;
                            order[static_cast<std::size_t>(pos)] = j;
                        }
#pragma omp parallel default(none) shared(offsets, order, w_blk, Xblk_norm, sum_x, sum_w) firstprivate(d, k)
                        {
                            std::vector<float> local_sum(static_cast<std::size_t>(d));
#pragma omp for schedule(static)
                            for (int c = 0; c < k; ++c) {
                                const int begin = offsets[static_cast<std::size_t>(c)];
                                const int end = offsets[static_cast<std::size_t>(c) + 1];
                                if (begin == end) continue;
                                std::fill(local_sum.begin(), local_sum.end(), 0.0f);
                                float wsum = 0.0f;
                                for (int p = begin; p < end; ++p) {
                                    const int j = order[static_cast<std::size_t>(p)];
                                    const float w = w_blk[static_cast<std::size_t>(j)];
                                    wsum += w;
                                    const float* xj = Xblk_norm.Col(j);
#pragma omp simd
                                    for (int r = 0; r < d; ++r) {
                                        local_sum[static_cast<std::size_t>(r)] += w * xj[r];
                                    }
                                }
                                float* dst = sum_x.Col(c);
#pragma omp simd
                                for (int r = 0; r < d; ++r) dst[r] += local_sum[static_cast<std::size_t>(r)];
                                sum_w[static_cast<std::size_t>(c)] += wsum;
                            }
                        }
                        acc2_s += t_acc.ElapsedSeconds();
                    }
                    if (overlap_h2d_pass2) {
                        cuda_kernels->KmeansMarkUploadSlotConsumed(overlap_slot_cur_pass2);
                        std::swap(overlap_slot_cur_pass2, overlap_slot_next_pass2);
                    }
                }
                if (!f_w_out) throw std::runtime_error("Failed to write: " + w_out_path);
                current_cost = (total_w > 0.0) ? static_cast<float>(total_cost / total_w) : 0.0f;

                f_w_out.flush();
                f_w_out.close();
                f_w_in.close();
                // Rotate weights for next iter.
                {
                    std::error_code ec;
                    std::filesystem::remove(w_in_path, ec);
                    ec.clear();
                    std::filesystem::rename(w_out_path, w_in_path, ec);
                    if (ec) {
                        throw std::runtime_error("Failed to rotate weight files: " + ec.message());
                    }
                }

                if (cuda_kernels) {
                    cuda_kernels->KmeansDownloadSums(&sum_x, &sum_w);
                    cuda_kernels->Sync();
                }

                // Finalize centers (same as spill path).
                Timer t_fin2;
#pragma omp parallel for default(none) shared(out, sum_w, fallback, X, sum_x) firstprivate(d, k, kEps) schedule(static)
                for (int c = 0; c < k; ++c) {
                    float* center = out.centers.Col(c);
                    const float w = sum_w[static_cast<std::size_t>(c)];
                    if (w <= kEps) {
                        const int idx = fallback[static_cast<std::size_t>(c)];
                        const float* xi = X.Col(idx);
                        float norm = 0.0f;
#pragma omp simd reduction(+:norm)
                        for (int r = 0; r < d; ++r) norm += xi[r] * xi[r];
                        norm = std::sqrt(std::max(norm, kEps));
                        const float inv = (norm > kEps) ? (1.0f / norm) : 1.0f;
#pragma omp simd
                        for (int r = 0; r < d; ++r) center[r] = xi[r] * inv;
                        continue;
                    }
                    const float inv_w = 1.0f / w;
                    const float* sx = sum_x.Col(c);
                    float norm = 0.0f;
                    for (int r = 0; r < d; ++r) {
                        const float v = sx[r] * inv_w;
                        center[r] = v;
                        norm += v * v;
                    }
                    norm = std::sqrt(std::max(norm, kEps));
                    const float inv = 1.0f / norm;
#pragma omp simd
                    for (int r = 0; r < d; ++r) center[r] *= inv;
                }
                const double fin2_s = t_fin2.ElapsedSeconds();
                if (profile_timing && timing) {
                    timing->accumulate_s += acc2_s;
                    timing->finalize_s += fin2_s;
                    timing->update_s += (acc2_s + fin2_s);
                }

                if (std::abs(prev_cost - current_cost) < cfg.tol) {
                    break;
                }
                prev_cost = current_cost;
                continue;
            }

            const std::string tmp_dir = !cfg.tmp_dir.empty() ? cfg.tmp_dir : std::string("./tmp");
            std::filesystem::create_directories(tmp_dir);
            const std::string costs_path = tmp_dir + "/kmeans_costs_f32.bin";
            const std::string assign_u16_path = tmp_dir + "/kmeans_assign_u16.bin";
            const std::string assign_u32_path = tmp_dir + "/kmeans_assign_u32.bin";
            const std::string w_in_path = tmp_dir + "/kmeans_weights_in_f32.bin";
            const std::string w_out_path = tmp_dir + "/kmeans_weights_out_f32.bin";

            // weights_in is always initialized at function entry (no caching).

            const bool use_u16_assign = (k <= 65535);
            {
                std::ofstream f_cost(costs_path, std::ios::binary | std::ios::trunc);
                if (!f_cost) throw std::runtime_error("Failed to open: " + costs_path);
                std::ofstream f_a16;
                std::ofstream f_a32;
                if (use_u16_assign) {
                    f_a16.open(assign_u16_path, std::ios::binary | std::ios::trunc);
                    if (!f_a16) throw std::runtime_error("Failed to open: " + assign_u16_path);
                }
                else {
                    f_a32.open(assign_u32_path, std::ios::binary | std::ios::trunc);
                    if (!f_a32) throw std::runtime_error("Failed to open: " + assign_u32_path);
                }

                std::vector<std::uint64_t> hist(static_cast<std::size_t>(std::max(1, cfg.quantile_bins)), 0);
                const int bins = std::max(1, cfg.quantile_bins);
                const float lo = 0.0f;
                const float hi = 2.0f;
                const float inv_span = static_cast<float>(bins) / (hi - lo);
                std::vector<float> cost_blk;
                std::vector<std::uint16_t> assign_u16_blk;
                std::vector<std::uint32_t> assign_u32_blk;

                const bool overlap_h2d_pass1 =
                    (cuda_kernels != nullptr) && (!have_full_xnorm) && (block_cols >= 4096);
                int overlap_slot_cur_pass1 = 0;
                DeviceMatF32View overlap_Xd_pass1[2];
                if (overlap_h2d_pass1 && n > 0) {
                    const int cols0 = std::min(block_cols, n);
                    const float* xptr0 = (X_pinned_base ? X_pinned_base : X.data.data());
                    overlap_Xd_pass1[0] = cuda_kernels->KmeansUploadAndNormalizeHostPtrBAsync(
                        /*slot=*/0,
                                 xptr0, /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols0,
                                 /*stage_to_pinned=*/(X_pinned_base == nullptr),
                                 /*stage_s=*/nullptr,
                                 /*h2d_s=*/nullptr,
                                 /*kernel_s=*/nullptr);
                    overlap_slot_cur_pass1 = 0;
                }

                for (int i0 = 0; i0 < n; i0 += block_cols) {
                    const int cols = std::min(block_cols, n - i0);
                    if (cuda_kernels) {
                        // GPU normalize + tiled GEMM + argmax (bounded score memory).
                        DeviceMatF32View Xd;
                        int slot_used = -1;
                        if (have_full_xnorm && Xnorm_full.ptr && Xnorm_full.rows == d && Xnorm_full.cols == n) {
                            Xd.ptr = Xnorm_full.ptr +
                                static_cast<std::size_t>(i0) * static_cast<std::size_t>(Xnorm_full.ld);
                            Xd.rows = d;
                            Xd.cols = cols;
                            Xd.ld = Xnorm_full.ld;
                        }
                        else {
                            if (overlap_h2d_pass1) {
                                const int slot_this = overlap_slot_cur_pass1;
                                const int slot_next = slot_this ^ 1;
                                const int i0_next = i0 + block_cols;
                                if (i0_next < n) {
                                    const int cols_next = std::min(block_cols, n - i0_next);
                                    const float* xptr_next =
                                        (X_pinned_base ? X_pinned_base : X.data.data()) +
                                        static_cast<std::size_t>(i0_next) * static_cast<std::size_t>(d);
                                    overlap_Xd_pass1[slot_next] = cuda_kernels->KmeansUploadAndNormalizeHostPtrBAsync(
                                        slot_next,
                                        xptr_next, /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols_next,
                                        /*stage_to_pinned=*/(X_pinned_base == nullptr),
                                        /*stage_s=*/nullptr,
                                        /*h2d_s=*/nullptr,
                                        /*kernel_s=*/nullptr);
                                }
                                cuda_kernels->KmeansComputeWaitForUpload(slot_this);
                                Xd = overlap_Xd_pass1[slot_this];
                                overlap_slot_cur_pass1 = slot_next;
                                slot_used = slot_this;
                            }
                            else {
                                Timer t_norm;
                                const float* xptr = (X_pinned_base ? X_pinned_base : X.data.data()) +
                                    static_cast<std::size_t>(i0) * static_cast<std::size_t>(d);
                                double stage_s = 0.0;
                                double h2d_s = 0.0;
                                double ker_s = 0.0;
                                if (profile_timing && timing) {
                                    Xd = cuda_kernels->KmeansUploadAndNormalizeHostPtrBTimed(
                                        xptr, /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols,
                                        /*stage_to_pinned=*/(X_pinned_base == nullptr),
                                        /*stage_s=*/&stage_s, &h2d_s, &ker_s);
                                }
                                else {
                                    Xd = cuda_kernels->KmeansUploadAndNormalizeHostPtrBTimed(
                                        xptr, /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols,
                                        /*stage_to_pinned=*/(X_pinned_base == nullptr),
                                        /*stage_s=*/nullptr,
                                        /*h2d_s=*/nullptr,
                                        /*kernel_s=*/nullptr);
                                }
                                if (profile_timing && timing) {
                                    timing->normalize_stage_s += stage_s;
                                    timing->normalize_h2d_s += h2d_s;
                                    timing->normalize_kernel_s += ker_s;
                                    timing->normalize_s += t_norm.ElapsedSeconds();
                                }
                            }
                        }

                        constexpr std::size_t kScoreBudgetBytes = 256ull * 1024ull * 1024ull; // 256 MiB
                        int tile_n = static_cast<int>(kScoreBudgetBytes /
                            (sizeof(float) * static_cast<std::size_t>(std::max(1, k))));
                        tile_n = std::max(1, tile_n);
                        tile_n = std::min(tile_n, cols);
                        if (tile_n > 256) tile_n = (tile_n / 256) * 256;
                        tile_n = std::max(1, tile_n);

                        cuda_kernels->KmeansEnsureBlockArgmax(cols);
                        DeviceVecI32View d_assign_tile;
                        DeviceVecF32View d_best_tile;
                        for (int j0 = 0; j0 < cols; j0 += tile_n) {
                            const int tcols = std::min(tile_n, cols - j0);
                            const float* dX = Xd.ptr + static_cast<std::size_t>(j0) * static_cast<std::size_t>(Xd.ld);

                            DeviceMatF32View scores;
                            Timer t_gemm;
                            stream_kernels->GemmDeviceDevicePtrB(/*transA=*/true, /*transB=*/false,
                                                                            /*alpha=*/1.0f,
                                                                            out.centers,
                                                                            dX,
                                                                            /*ldB=*/Xd.ld,
                                                                            /*rowsB=*/Xd.rows,
                                                                            /*colsB=*/tcols,
                                                                            /*beta=*/0.0f,
                                                                            &scores);
                            if (profile_timing && timing) {
                                stream_kernels->SyncCompute();
                                timing->assign_gemm_s += t_gemm.ElapsedSeconds();
                            }

                            Timer t_arg;
                            stream_kernels->ArgmaxColsF32Device(scores, &d_assign_tile, &d_best_tile);
                            cuda_kernels->KmeansCopyTileArgmaxToBlock(d_assign_tile, d_best_tile, j0);
                            if (profile_timing && timing) {
                                stream_kernels->SyncCompute();
                                timing->assign_argmax_s += t_arg.ElapsedSeconds();
                            }
                        }
                        cuda_kernels->KmeansDownloadBlockArgmax(&assign_blk, &dot_blk);
                        if (slot_used >= 0) {
                            cuda_kernels->KmeansMarkUploadSlotConsumed(slot_used);
                        }
                    }
                    else {
                        if (Xblk_norm.cols != cols) {
                            Xblk_norm = ColMajorMatrix<float>(d, cols);
                        }

                        if (profile_timing && timing) {
                            Timer t_norm;
#pragma omp parallel for default(none) shared(X, Xblk_norm) firstprivate(i0, cols, d, kEps) schedule(static)
                            for (int j = 0; j < cols; ++j) {
                                const float* xi = X.Col(i0 + j);
                                float* dst = Xblk_norm.Col(j);
                                float norm = 0.0f;
#pragma omp simd reduction(+:norm)
                                for (int r = 0; r < d; ++r) norm += xi[r] * xi[r];
                                norm = std::sqrt(std::max(norm, kEps));
                                const float inv = (norm > kEps) ? (1.0f / norm) : 1.0f;
#pragma omp simd
                                for (int r = 0; r < d; ++r) dst[r] = xi[r] * inv;
                            }
                            timing->normalize_s += t_norm.ElapsedSeconds();
                        }
                        else {
#pragma omp parallel for default(none) shared(X, Xblk_norm) firstprivate(i0, cols, d, kEps) schedule(static)
                            for (int j = 0; j < cols; ++j) {
                                const float* xi = X.Col(i0 + j);
                                float* dst = Xblk_norm.Col(j);
                                float norm = 0.0f;
#pragma omp simd reduction(+:norm)
                                for (int r = 0; r < d; ++r) norm += xi[r] * xi[r];
                                norm = std::sqrt(std::max(norm, kEps));
                                const float inv = (norm > kEps) ? (1.0f / norm) : 1.0f;
#pragma omp simd
                                for (int r = 0; r < d; ++r) dst[r] = xi[r] * inv;
                            }
                        }

                        AssignSamplesGpu(Xblk_norm, out.centers, stream_kernels,
                                         profile_timing, timing,
                                         &assign_blk, &dot_blk);
                    }
                    cost_blk.resize(static_cast<std::size_t>(cols));
                    if (use_u16_assign) {
                        assign_u16_blk.resize(static_cast<std::size_t>(cols));
                    }
                    else {
                        assign_u32_blk.resize(static_cast<std::size_t>(cols));
                    }
                    for (int j = 0; j < cols; ++j) {
                        const int c = assign_blk[static_cast<std::size_t>(j)];
                        out.assignments[static_cast<std::size_t>(i0 + j)] = c;
                        const float cost = 1.0f - dot_blk[static_cast<std::size_t>(j)];
                        cost_blk[static_cast<std::size_t>(j)] = cost;
                        if (use_u16_assign) {
                            assign_u16_blk[static_cast<std::size_t>(j)] = static_cast<std::uint16_t>(c);
                        }
                        else {
                            assign_u32_blk[static_cast<std::size_t>(j)] = static_cast<std::uint32_t>(c);
                        }
                        float x = cost;
                        if (x < lo) x = lo;
                        if (x > hi) x = hi;
                        int b = static_cast<int>((x - lo) * inv_span);
                        if (b >= bins) b = bins - 1;
                        hist[static_cast<std::size_t>(b)] += 1;
                    }
                    Timer t_spill;
                    f_cost.write(reinterpret_cast<const char*>(cost_blk.data()),
                                 static_cast<std::streamsize>(sizeof(float) * cost_blk.size()));
                    if (use_u16_assign) {
                        f_a16.write(reinterpret_cast<const char*>(assign_u16_blk.data()),
                                    static_cast<std::streamsize>(sizeof(std::uint16_t) * assign_u16_blk.size()));
                    }
                    else {
                        f_a32.write(reinterpret_cast<const char*>(assign_u32_blk.data()),
                                    static_cast<std::streamsize>(sizeof(std::uint32_t) * assign_u32_blk.size()));
                    }
                    if (profile_timing && timing) timing->spill_s += t_spill.ElapsedSeconds();
                }
                if (!f_cost) throw std::runtime_error("Failed to write: " + costs_path);
                if (use_u16_assign) {
                    if (!f_a16) throw std::runtime_error("Failed to write: " + assign_u16_path);
                }
                else {
                    if (!f_a32) throw std::runtime_error("Failed to write: " + assign_u32_path);
                }

                // Ensure pass-1 outputs are fully flushed before we reopen them for reading.
                f_cost.flush();
                if (use_u16_assign) {
                    f_a16.flush();
                }
                else {
                    f_a32.flush();
                }
                f_cost.close();
                if (use_u16_assign) {
                    f_a16.close();
                }
                else {
                    f_a32.close();
                }

                // Determinism: serial accumulation (same as ComputeCost) for stopping criterion.
                // Compute approx quantile threshold from histogram.
                const std::uint64_t target = static_cast<std::uint64_t>(
                    cfg.outlier_quantile * static_cast<float>(std::max(1, n - 1)));
                std::uint64_t prefix = 0;
                int bq = bins - 1;
                for (int b = 0; b < bins; ++b) {
                    prefix += hist[static_cast<std::size_t>(b)];
                    if (prefix > target) {
                        bq = b;
                        break;
                    }
                }
                const float bw = (hi - lo) / static_cast<float>(bins);
                const float quantile_threshold = lo + (static_cast<float>(bq) + 0.5f) * bw;
                const float effective_threshold = std::max(quantile_threshold, cfg.cost_threshold);

                // Pass 2a: update weights and compute current cost using streamed (costs, weights_in).
                float current_cost = 0.0f;
                std::vector<float> w_blk;
                {
                    Timer t_weights;
                    std::ifstream f_cost_in(costs_path, std::ios::binary);
                    if (!f_cost_in) throw std::runtime_error("Failed to open: " + costs_path);
                    std::ifstream f_w_in(w_in_path, std::ios::binary);
                    if (!f_w_in) throw std::runtime_error("Failed to open: " + w_in_path);
                    std::ofstream f_w_out(w_out_path, std::ios::binary | std::ios::trunc);
                    if (!f_w_out) throw std::runtime_error("Failed to open: " + w_out_path);

                    const float denom = static_cast<float>(std::max(1, cfg.max_iters - cfg.warmup_iters));
                    const float annealed_factor =
                        cfg.annealing_factor * (1.0f - std::exp(-(iter - cfg.warmup_iters) / denom));

                    double total_cost = 0.0;
                    double total_w = 0.0;
                    std::vector<float> cost_blk;
                    for (int i0 = 0; i0 < n; i0 += block_cols) {
                        const int cols = std::min(block_cols, n - i0);
                        ReadVectorExactOrThrow(f_w_in, &w_blk, static_cast<std::size_t>(cols));
                        ReadVectorExactOrThrow(f_cost_in, &cost_blk, static_cast<std::size_t>(cols));
                        for (int j = 0; j < cols; ++j) {
                            const float cost = cost_blk[static_cast<std::size_t>(j)];
                            float w = w_blk[static_cast<std::size_t>(j)];
                            if (cost > effective_threshold) {
                                w = std::max(cfg.min_weight, w * (1.0f - annealed_factor));
                            }
                            else {
                                w = std::min(1.0f, w * (1.0f + annealed_factor * 0.5f));
                            }
                            w_blk[static_cast<std::size_t>(j)] = w;
                            total_cost += static_cast<double>(w) * static_cast<double>(cost);
                            total_w += static_cast<double>(w);
                        }
                        f_w_out.write(reinterpret_cast<const char*>(w_blk.data()),
                                      static_cast<std::streamsize>(sizeof(float) * w_blk.size()));
                    }
                    if (!f_w_out) throw std::runtime_error("Failed to write: " + w_out_path);
                    current_cost = (total_w > 0.0) ? static_cast<float>(total_cost / total_w) : 0.0f;
                    if (profile_timing && timing) timing->weights_s += t_weights.ElapsedSeconds();
                } // close weight/cost streams before rotating files

                // Update adaptive weights from costs.
                if (profile_timing && timing) {
                    // Approx weights time: include histogram + weight pass (already done).
                    // This will be filled by the caller if needed; avoid extra timers here.
                }

                // Pass 2: update centers from raw X using assignments and updated weights.
                double acc2_s = 0.0;
                // Swap-in updated weights file for accumulation.
                std::error_code ec;
                std::filesystem::remove(w_in_path, ec);
                ec.clear();
                std::filesystem::rename(w_out_path, w_in_path, ec);
                if (ec) {
                    throw std::runtime_error("Failed to rotate weight files: " + ec.message());
                }
                // Accumulate using weights read from file in a streaming manner.
                {
                    ColMajorMatrix<float> sum_x(d, k);
                    std::vector<float> sum_w(static_cast<std::size_t>(k), 0.0f);
                    std::ifstream f_w(w_in_path, std::ios::binary);
                    if (!f_w) throw std::runtime_error("Failed to open: " + w_in_path);

                    // GPU path: normalize + weighted accumulation on device, then download sums.
                    if (cuda_kernels) {
                        cuda_kernels->KmeansResetSums(d, k);
                    }
                    else {
                        // Reuse bucket scratch buffers across blocks.
                        // (CPU path only.)
                    }
                    std::vector<int> offsets(static_cast<std::size_t>(k) + 1, 0);
                    std::vector<int> cursor(static_cast<std::size_t>(k) + 1, 0);
                    std::vector<int> order;
                    std::vector<int> assign_blk2;

                    const bool overlap_h2d_pass2 =
                        (cuda_kernels != nullptr) && (!have_full_xnorm) && (block_cols >= 4096);
                    int overlap_slot_cur_pass2 = 0;
                    DeviceMatF32View overlap_Xd_pass2[2];
                    if (overlap_h2d_pass2 && n > 0) {
                        const int cols0 = std::min(block_cols, n);
                        const float* xptr0 = (X_pinned_base ? X_pinned_base : X.data.data());
                        overlap_Xd_pass2[0] = cuda_kernels->KmeansUploadAndNormalizeHostPtrBAsync(
                            /*slot=*/0,
                                     xptr0, /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols0,
                                     /*stage_to_pinned=*/(X_pinned_base == nullptr),
                                     /*stage_s=*/nullptr,
                                     /*h2d_s=*/nullptr,
                                     /*kernel_s=*/nullptr);
                        overlap_slot_cur_pass2 = 0;
                    }

                    for (int i0 = 0; i0 < n; i0 += block_cols) {
                        const int cols = std::min(block_cols, n - i0);
                        if (profile_timing && timing) {
                            Timer t_wread;
                            ReadVectorExactOrThrow(f_w, &w_blk, static_cast<std::size_t>(cols));
                            timing->update_read_weights_s += t_wread.ElapsedSeconds();
                        }
                        else {
                            ReadVectorExactOrThrow(f_w, &w_blk, static_cast<std::size_t>(cols));
                        }

                        if (cuda_kernels) {
                            // Build assignments for this block from the stored global assignments.
                            assign_blk2.resize(static_cast<std::size_t>(cols));
                            for (int j = 0; j < cols; ++j) {
                                assign_blk2[static_cast<std::size_t>(j)] =
                                    out.assignments[static_cast<std::size_t>(i0 + j)];
                            }

                            DeviceMatF32View Xd;
                            int slot_used = -1;
                            if (have_full_xnorm && Xnorm_full.ptr && Xnorm_full.rows == d && Xnorm_full.cols == n) {
                                Xd.ptr = Xnorm_full.ptr +
                                    static_cast<std::size_t>(i0) * static_cast<std::size_t>(Xnorm_full.ld);
                                Xd.rows = d;
                                Xd.cols = cols;
                                Xd.ld = Xnorm_full.ld;
                            }
                            else {
                                if (overlap_h2d_pass2) {
                                    const int slot_this = overlap_slot_cur_pass2;
                                    const int slot_next = slot_this ^ 1;
                                    const int i0_next = i0 + block_cols;
                                    if (i0_next < n) {
                                        const int cols_next = std::min(block_cols, n - i0_next);
                                        const float* xptr_next =
                                            (X_pinned_base ? X_pinned_base : X.data.data()) +
                                            static_cast<std::size_t>(i0_next) * static_cast<std::size_t>(d);
                                        overlap_Xd_pass2[slot_next] = cuda_kernels->
                                            KmeansUploadAndNormalizeHostPtrBAsync(
                                                slot_next,
                                                xptr_next, /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols_next,
                                                /*stage_to_pinned=*/(X_pinned_base == nullptr),
                                                /*stage_s=*/nullptr,
                                                /*h2d_s=*/nullptr,
                                                /*kernel_s=*/nullptr);
                                    }
                                    cuda_kernels->KmeansComputeWaitForUpload(slot_this);
                                    Xd = overlap_Xd_pass2[slot_this];
                                    overlap_slot_cur_pass2 = slot_next;
                                    slot_used = slot_this;
                                }
                                else {
                                    Timer t_norm2;
                                    const float* xptr = (X_pinned_base ? X_pinned_base : X.data.data()) +
                                        static_cast<std::size_t>(i0) * static_cast<std::size_t>(d);
                                    double stage_s = 0.0;
                                    double h2d_s = 0.0;
                                    double ker_s = 0.0;
                                    if (profile_timing && timing) {
                                        Xd = cuda_kernels->KmeansUploadAndNormalizeHostPtrBTimed(
                                            xptr, /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols,
                                            /*stage_to_pinned=*/(X_pinned_base == nullptr),
                                            /*stage_s=*/&stage_s, &h2d_s, &ker_s);
                                    }
                                    else {
                                        Xd = cuda_kernels->KmeansUploadAndNormalizeHostPtrBTimed(
                                            xptr, /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols,
                                            /*stage_to_pinned=*/(X_pinned_base == nullptr),
                                            /*stage_s=*/nullptr,
                                            /*h2d_s=*/nullptr,
                                            /*kernel_s=*/nullptr);
                                    }
                                    if (profile_timing && timing) {
                                        timing->normalize_stage_s += stage_s;
                                        timing->normalize_h2d_s += h2d_s;
                                        timing->normalize_kernel_s += ker_s;
                                        timing->normalize_s += t_norm2.ElapsedSeconds();
                                    }
                                }
                            }

                            Timer t_acc;
                            cuda_kernels->KmeansAccumulateWeights(Xd, assign_blk2, w_blk);
                            if (profile_timing && timing) {
                                cuda_kernels->SyncCompute();
                            }
                            acc2_s += t_acc.ElapsedSeconds();
                            if (slot_used >= 0) {
                                cuda_kernels->KmeansMarkUploadSlotConsumed(slot_used);
                            }
                        }
                        else {
                            if (Xblk_norm.cols != cols) {
                                Xblk_norm = ColMajorMatrix<float>(d, cols);
                            }

                            // Normalize this block.
                            if (profile_timing && timing) {
                                Timer t_norm2;
#pragma omp parallel for default(none) shared(X, Xblk_norm) firstprivate(i0, cols, d, kEps) schedule(static)
                                for (int j = 0; j < cols; ++j) {
                                    const float* xi = X.Col(i0 + j);
                                    float* dst = Xblk_norm.Col(j);
                                    float norm = 0.0f;
#pragma omp simd reduction(+:norm)
                                    for (int r = 0; r < d; ++r) norm += xi[r] * xi[r];
                                    norm = std::sqrt(std::max(norm, kEps));
                                    const float inv = (norm > kEps) ? (1.0f / norm) : 1.0f;
#pragma omp simd
                                    for (int r = 0; r < d; ++r) dst[r] = xi[r] * inv;
                                }
                                timing->normalize_s += t_norm2.ElapsedSeconds();
                            }
                            else {
#pragma omp parallel for default(none) shared(X, Xblk_norm) firstprivate(i0, cols, d, kEps) schedule(static)
                                for (int j = 0; j < cols; ++j) {
                                    const float* xi = X.Col(i0 + j);
                                    float* dst = Xblk_norm.Col(j);
                                    float norm = 0.0f;
#pragma omp simd reduction(+:norm)
                                    for (int r = 0; r < d; ++r) norm += xi[r] * xi[r];
                                    norm = std::sqrt(std::max(norm, kEps));
                                    const float inv = (norm > kEps) ? (1.0f / norm) : 1.0f;
#pragma omp simd
                                    for (int r = 0; r < d; ++r) dst[r] = xi[r] * inv;
                                }
                            }

                            Timer t_acc;
                            // Bucket within this block (per-cluster contiguous ranges).
                            std::fill(offsets.begin(), offsets.end(), 0);
                            for (int j = 0; j < cols; ++j) {
                                const int c = out.assignments[static_cast<std::size_t>(i0 + j)];
                                ++offsets[static_cast<std::size_t>(c) + 1];
                            }
                            for (int c = 0; c < k; ++c) {
                                offsets[static_cast<std::size_t>(c) + 1] += offsets[static_cast<std::size_t>(c)];
                            }
                            order.resize(static_cast<std::size_t>(cols));
                            std::copy(offsets.begin(), offsets.end(), cursor.begin());
                            for (int j = 0; j < cols; ++j) {
                                const int c = out.assignments[static_cast<std::size_t>(i0 + j)];
                                const int pos = cursor[static_cast<std::size_t>(c)]++;
                                order[static_cast<std::size_t>(pos)] = j;
                            }

#pragma omp parallel default(none) shared(offsets, order, w_blk, Xblk_norm, sum_x, sum_w) firstprivate(d, k)
                            {
                                std::vector<float> local_sum(static_cast<std::size_t>(d));
#pragma omp for schedule(static)
                                for (int c = 0; c < k; ++c) {
                                    const int begin = offsets[static_cast<std::size_t>(c)];
                                    const int end = offsets[static_cast<std::size_t>(c) + 1];
                                    if (begin == end) continue;

                                    std::fill(local_sum.begin(), local_sum.end(), 0.0f);
                                    float wsum = 0.0f;
                                    for (int p = begin; p < end; ++p) {
                                        const int j = order[static_cast<std::size_t>(p)];
                                        const float w = w_blk[static_cast<std::size_t>(j)];
                                        wsum += w;
                                        const float* xj = Xblk_norm.Col(j);
#pragma omp simd
                                        for (int r = 0; r < d; ++r) {
                                            local_sum[static_cast<std::size_t>(r)] += w * xj[r];
                                        }
                                    }
                                    float* dst = sum_x.Col(c);
#pragma omp simd
                                    for (int r = 0; r < d; ++r) dst[r] += local_sum[static_cast<std::size_t>(r)];
                                    sum_w[static_cast<std::size_t>(c)] += wsum;
                                }
                            }
                            acc2_s += t_acc.ElapsedSeconds();
                        }
                    }

                    if (cuda_kernels) {
                        cuda_kernels->KmeansDownloadSums(&sum_x, &sum_w);
                        cuda_kernels->Sync();
                    }

                    // Finalize centers.
                    Timer t_fin2;
#pragma omp parallel for default(none) shared(out, sum_w, fallback, X, sum_x) firstprivate(d, k, kEps) schedule(static)
                    for (int c = 0; c < k; ++c) {
                        float* center = out.centers.Col(c);
                        const float w = sum_w[static_cast<std::size_t>(c)];
                        if (w <= kEps) {
                            const int idx = fallback[static_cast<std::size_t>(c)];
                            const float* xi = X.Col(idx);
                            float norm = 0.0f;
#pragma omp simd reduction(+:norm)
                            for (int r = 0; r < d; ++r) norm += xi[r] * xi[r];
                            norm = std::sqrt(std::max(norm, kEps));
                            const float inv = (norm > kEps) ? (1.0f / norm) : 1.0f;
#pragma omp simd
                            for (int r = 0; r < d; ++r) center[r] = xi[r] * inv;
                            continue;
                        }
                        const float inv_w = 1.0f / w;
                        const float* sx = sum_x.Col(c);
                        float norm = 0.0f;
                        for (int r = 0; r < d; ++r) {
                            const float v = sx[r] * inv_w;
                            center[r] = v;
                            norm += v * v;
                        }
                        norm = std::sqrt(std::max(norm, kEps));
                        const float inv = 1.0f / norm;
#pragma omp simd
                        for (int r = 0; r < d; ++r) center[r] *= inv;
                    }
                    const double fin2_s = t_fin2.ElapsedSeconds();
                    if (profile_timing && timing) {
                        timing->accumulate_s += acc2_s;
                        timing->finalize_s += fin2_s;
                        timing->update_s += (acc2_s + fin2_s);
                    }
                }

                if (std::abs(prev_cost - current_cost) < cfg.tol) {
                    break;
                }
                prev_cost = current_cost;
            }
        }

        if (timing) timing->total_s += t_total.ElapsedSeconds();
        if (cuda_kernels) {
            // Avoid holding a large X_norm cache across later pipeline phases (beam/ICM/etc.).
            cuda_kernels->KmeansReleasePartialXnormCache();
            cuda_kernels->KmeansReleaseXnormCache();
        }
        return out;
    }

    ColMajorMatrix<float> SphericalKmeansCentersOnlyStreamingReaderRvqResidualGpu(
        io::IColBlockReader* reader,
        int k,
        const KmeansConfig& cfg,
        std::mt19937* rng,
        StreamKernelProvider* stream_kernels,
        const RvqInitCodesInMemory* prior_codes,
        const std::vector<ColMajorMatrix<float>>* prior_codebooks,
        int upto_layer,
        bool profile_timing,
        KmeansTiming* timing,
        std::string* err) {
        if (upto_layer <= 0) {
            return SphericalKmeansCentersOnlyStreamingReader(reader, k, cfg, rng, stream_kernels,
                                                             profile_timing, timing, err);
        }
        if (!prior_codes || !prior_codebooks) {
            if (err) *err = "SphericalKmeansCentersOnlyStreamingReaderRvqResidualGpu: prior_codes/codebooks are null.";
            return {};
        }
        if (upto_layer > prior_codes->m() || upto_layer > static_cast<int>(prior_codebooks->size())) {
            if (err) *err = "SphericalKmeansCentersOnlyStreamingReaderRvqResidualGpu: upto_layer out of range.";
            return {};
        }
        if (!reader) {
            if (err) *err = "SphericalKmeansCentersOnlyStreamingReaderRvqResidualGpu: reader is null.";
            return {};
        }
        if (!stream_kernels || !stream_kernels->IsGpu()) {
            if (err) *err = "SphericalKmeansCentersOnlyStreamingReaderRvqResidualGpu requires CUDA stream_kernels.";
            return {};
        }
        if (prior_codes->n() != reader->n()) {
            if (err) *err = "SphericalKmeansCentersOnlyStreamingReaderRvqResidualGpu: prior_codes.n != reader.n.";
            return {};
        }

        RvqResidualHook hook;
        hook.codes = prior_codes;
        hook.codebooks = prior_codebooks;
        hook.upto_layer = upto_layer;
        ScopedRvqResidualHook scoped(&hook);
        return SphericalKmeansCentersOnlyStreamingReader(reader, k, cfg, rng, stream_kernels,
                                                         profile_timing, timing, err);
    }

    ColMajorMatrix<float> SphericalKmeansCentersOnlyStreamingReaderEmitCodes(io::IColBlockReader* reader,
                                                                             int k,
                                                                             const KmeansConfig& cfg,
                                                                             std::mt19937* rng,
                                                                             StreamKernelProvider* stream_kernels,
                                                                             RvqInitCodesInMemory* codes_out,
                                                                             int out_layer,
                                                                             bool profile_timing,
                                                                             KmeansTiming* timing,
                                                                             std::string* err) {
        RvqResidualHook hook;
        hook.codes = nullptr;
        hook.codebooks = nullptr;
        hook.upto_layer = 0;
        hook.emit_codes = codes_out;
        hook.emit_layer = out_layer;
        ScopedRvqResidualHook scoped(&hook);
        return SphericalKmeansCentersOnlyStreamingReader(reader, k, cfg, rng, stream_kernels,
                                                         profile_timing, timing, err);
    }

    ColMajorMatrix<float> SphericalKmeansCentersOnlyStreamingReaderRvqResidualGpuEmitCodes(
        io::IColBlockReader* reader,
        int k,
        const KmeansConfig& cfg,
        std::mt19937* rng,
        StreamKernelProvider* stream_kernels,
        const RvqInitCodesInMemory* prior_codes,
        const std::vector<ColMajorMatrix<float>>* prior_codebooks,
        int upto_layer,
        RvqInitCodesInMemory* codes_out,
        int out_layer,
        bool profile_timing,
        KmeansTiming* timing,
        std::string* err) {
        if (upto_layer <= 0) {
            return SphericalKmeansCentersOnlyStreamingReaderEmitCodes(reader, k, cfg, rng, stream_kernels,
                                                                      codes_out, out_layer,
                                                                      profile_timing, timing, err);
        }
        if (!prior_codes || !prior_codebooks) {
            if (err)
                *err =
                    "SphericalKmeansCentersOnlyStreamingReaderRvqResidualGpuEmitCodes: prior_codes/codebooks are null.";
            return {};
        }
        if (upto_layer > prior_codes->m() || upto_layer > static_cast<int>(prior_codebooks->size())) {
            if (err)
                *err =
                    "SphericalKmeansCentersOnlyStreamingReaderRvqResidualGpuEmitCodes: upto_layer out of range.";
            return {};
        }
        if (!reader) {
            if (err) *err = "SphericalKmeansCentersOnlyStreamingReaderRvqResidualGpuEmitCodes: reader is null.";
            return {};
        }
        if (!stream_kernels || !stream_kernels->IsGpu()) {
            if (err)
                *err =
                    "SphericalKmeansCentersOnlyStreamingReaderRvqResidualGpuEmitCodes requires CUDA stream_kernels.";
            return {};
        }
        if (prior_codes->n() != reader->n()) {
            if (err)
                *err =
                    "SphericalKmeansCentersOnlyStreamingReaderRvqResidualGpuEmitCodes: prior_codes.n != reader.n.";
            return {};
        }

        RvqResidualHook hook;
        hook.codes = prior_codes;
        hook.codebooks = prior_codebooks;
        hook.upto_layer = upto_layer;
        hook.emit_codes = codes_out;
        hook.emit_layer = out_layer;
        ScopedRvqResidualHook scoped(&hook);
        return SphericalKmeansCentersOnlyStreamingReader(reader, k, cfg, rng, stream_kernels,
                                                         profile_timing, timing, err);
    }
} // namespace stlq
