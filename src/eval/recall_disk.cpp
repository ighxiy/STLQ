#include "stlq/eval/recall_disk.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <omp.h>

#include "stlq/core/blas.h"
#include "stlq/core/threading.h"
#include "stlq/eval/norm2_lut.h"
#include "stlq/common/types.h"
#include "stlq/ivf/cluster_select.h"
#include "stlq/common/logger.h"
#include "stlq/quantizer/linear_algebra.h"
#include "stlq/quantizer/precomp_large_root.h"
#include "stlq/common/timer.h"

// ---- AVX2 SIMD support for ADC scan acceleration ----
#if (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)) && !defined(STLQ_NO_AVX2)
#define STLQ_X86_SIMD 1
#include <immintrin.h>
#ifndef _MSC_VER
static bool DetectAvx2Fma() {
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}
// GCC/Clang: per-function target attribute enables AVX2/FMA codegen without global -mavx2.
#define STLQ_AVX2_FMA_TARGET __attribute__((target("avx2,fma")))
#else
#include <intrin.h>
static bool DetectAvx2Fma() {
    int info[4] = {};
    __cpuidex(info, 7, 0);
    bool avx2  = (info[1] >> 5) & 1;
    __cpuid(info, 1);
    bool fma   = (info[2] >> 12) & 1;
    return avx2 && fma;
}
// MSVC: intrinsics are always available when <immintrin.h> is included;
// no per-function target attribute needed (or supported).
#define STLQ_AVX2_FMA_TARGET
#endif
static const bool g_has_avx2_fma = DetectAvx2Fma();
#endif  // STLQ_X86_SIMD

namespace stlq {

namespace {

float ComputeRecallAtKMatrix(const std::vector<int>& ground_truth,
                             const ColMajorMatrix<int>& indices) {
    const int k = indices.rows;
    const int nquery = indices.cols;
    if (k <= 0 || nquery <= 0) {
        return 0.0f;
    }
    if (ground_truth.size() != static_cast<std::size_t>(nquery)) {
        return 0.0f;
    }
    int hits = 0;
    for (int qi = 0; qi < nquery; ++qi) {
        const int g = ground_truth[static_cast<std::size_t>(qi)];
        const int* row = indices.Col(qi);
        for (int j = 0; j < k; ++j) {
            if (row[j] == g) {
                ++hits;
                break;
            }
        }
    }
    return static_cast<float>(hits) / static_cast<float>(nquery);
}

// Deterministic ordering: (dist, id). Better = smaller dist; ties => smaller id.
struct ByDistThenIdLess {
    bool operator()(const std::pair<float, int32_t>& a,
                    const std::pair<float, int32_t>& b) const {
        if (a.first != b.first) {
            return a.first < b.first;
        }
        return a.second < b.second;
    }
};

struct Candidate {
    float dist = 0.0f;
    std::int32_t id = -1;
};

STLQ_ALWAYS_INLINE bool CandLess(const Candidate& a, const Candidate& b) {
    if (a.dist < b.dist) return true;
    if (a.dist > b.dist) return false;
    return a.id < b.id;
}

struct TopKHeap {
    int k = 0;
    std::vector<Candidate> heap;  // max-heap by CandLess (so heap[0] is current "worst")
    bool heapified = false;

    explicit TopKHeap(int k_in) : k(std::max(1, k_in)) {
        heap.reserve(static_cast<std::size_t>(k));
    }

    STLQ_ALWAYS_INLINE void SiftDown(int i) {
        const int n = static_cast<int>(heap.size());
        Candidate x = heap[static_cast<std::size_t>(i)];
        while (true) {
            const int l = i * 2 + 1;
            if (l >= n) break;
            int c = l;
            const int r = l + 1;
            if (r < n && CandLess(heap[static_cast<std::size_t>(c)],
                                  heap[static_cast<std::size_t>(r)])) {
                c = r;
            }
            if (!CandLess(x, heap[static_cast<std::size_t>(c)])) break;
            heap[static_cast<std::size_t>(i)] = heap[static_cast<std::size_t>(c)];
            i = c;
        }
        heap[static_cast<std::size_t>(i)] = x;
    }

    STLQ_ALWAYS_INLINE void Heapify() {
        for (int i = (static_cast<int>(heap.size()) >> 1) - 1; i >= 0; --i) {
            SiftDown(i);
        }
        heapified = true;
    }

    STLQ_ALWAYS_INLINE void Push(float dist, std::int32_t id) {
        const Candidate x{dist, id};
        if (static_cast<int>(heap.size()) < k) {
            heap.push_back(x);
            if (static_cast<int>(heap.size()) == k) {
                Heapify();
            }
            return;
        }
        if (!heapified) {
            Heapify();
        }
        const Candidate& worst = heap[0];
        if (x.dist > worst.dist || (x.dist == worst.dist && x.id >= worst.id)) return;
        heap[0] = x;
        SiftDown(0);
    }

    // Assumes: heap.size()==k, heapified==true, and (dist,id) is strictly better than current worst.
    STLQ_ALWAYS_INLINE void PushAssumeBetter(float dist, std::int32_t id) {
        heap[0] = Candidate{dist, id};
        SiftDown(0);
    }

    void Finalize(float* out_dists, int* out_ids) {
        std::sort(heap.begin(), heap.end(), [](const Candidate& a, const Candidate& b) {
            return CandLess(a, b);
        });
        const int got = static_cast<int>(heap.size());
        for (int i = 0; i < got; ++i) {
            out_dists[i] = heap[static_cast<std::size_t>(i)].dist;
            out_ids[i] = static_cast<int>(heap[static_cast<std::size_t>(i)].id);
        }
        for (int i = got; i < k; ++i) {
            out_dists[i] = std::numeric_limits<float>::infinity();
            out_ids[i] = -1;
        }
    }
};

// ---- AVX2 batch-of-8 ADC scan functions (m=5, m=10) ----
// These compute 8 distances in parallel using gather + FMA, with batch rejection
// for the common case where all 8 candidates are worse than the current heap top.
#if defined(STLQ_X86_SIMD)

// Helper: push one (dist, gid) into the heap, updating worst/have_worst.
STLQ_ALWAYS_INLINE void HeapPushOne(float dist, std::int32_t gid,
                                       TopKHeap& heap, Candidate& worst, bool& have_worst) {
    if (static_cast<int>(heap.heap.size()) < heap.k) {
        heap.heap.push_back(Candidate{dist, gid});
        if (static_cast<int>(heap.heap.size()) == heap.k) {
            heap.Heapify();
            worst = heap.heap[0];
            have_worst = true;
        }
        return;
    }
    if (!have_worst) {
        heap.Heapify();
        worst = heap.heap[0];
        have_worst = true;
    }
    if (dist > worst.dist || (dist == worst.dist && gid >= worst.id)) return;
    heap.PushAssumeBetter(dist, gid);
    worst = heap.heap[0];
}

STLQ_AVX2_FMA_TARGET
static void AdcScanClusterM5_Avx2(
        const Code* __restrict codes,     // stride 4
        const float* __restrict coeffs,   // stride 5
        const float* __restrict norm2,
        const std::uint32_t* __restrict gids,
        const float* __restrict xsmall,
        float root_score,
        int off1, int off2, int off3, int off4,
        std::uint32_t len,
        TopKHeap& heap, Candidate& worst, bool& have_worst) {
    const __m256i stride5 = _mm256_setr_epi32(0, 5, 10, 15, 20, 25, 30, 35);
    const __m256i cidx1 = _mm256_add_epi32(stride5, _mm256_set1_epi32(1));
    const __m256i cidx2 = _mm256_add_epi32(stride5, _mm256_set1_epi32(2));
    const __m256i cidx3 = _mm256_add_epi32(stride5, _mm256_set1_epi32(3));
    const __m256i cidx4 = _mm256_add_epi32(stride5, _mm256_set1_epi32(4));
    const __m256i mask_ff = _mm256_set1_epi32(0xFF);
    const __m256 two = _mm256_set1_ps(2.0f);
    const __m256 root_v = _mm256_set1_ps(root_score);
    const __m256i off1_v = _mm256_set1_epi32(off1);
    const __m256i off2_v = _mm256_set1_epi32(off2);
    const __m256i off3_v = _mm256_set1_epi32(off3);
    const __m256i off4_v = _mm256_set1_epi32(off4);

    std::uint32_t j = 0;
    for (; j + 8 <= len; j += 8) {
        const __m256 norm_8 = _mm256_loadu_ps(norm2 + j);
        // 8 vectors × 4 codes = 32 contiguous bytes (each int32 = 4 code bytes for one vector).
        const __m256i packed = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(codes + static_cast<std::size_t>(j) * 4u));
        const float* cb = coeffs + static_cast<std::size_t>(j) * 5u;

        // a[0] * root_score
        __m256 acc = _mm256_mul_ps(_mm256_i32gather_ps(cb, stride5, 4), root_v);
        // Layer 1: code byte 0
        __m256i idx = _mm256_add_epi32(_mm256_and_si256(packed, mask_ff), off1_v);
        acc = _mm256_fmadd_ps(_mm256_i32gather_ps(cb, cidx1, 4),
                              _mm256_i32gather_ps(xsmall, idx, 4), acc);
        // Layer 2: code byte 1
        idx = _mm256_add_epi32(_mm256_and_si256(_mm256_srli_epi32(packed, 8), mask_ff), off2_v);
        acc = _mm256_fmadd_ps(_mm256_i32gather_ps(cb, cidx2, 4),
                              _mm256_i32gather_ps(xsmall, idx, 4), acc);
        // Layer 3: code byte 2
        idx = _mm256_add_epi32(_mm256_and_si256(_mm256_srli_epi32(packed, 16), mask_ff), off3_v);
        acc = _mm256_fmadd_ps(_mm256_i32gather_ps(cb, cidx3, 4),
                              _mm256_i32gather_ps(xsmall, idx, 4), acc);
        // Layer 4: code byte 3
        idx = _mm256_add_epi32(_mm256_srli_epi32(packed, 24), off4_v);
        acc = _mm256_fmadd_ps(_mm256_i32gather_ps(cb, cidx4, 4),
                              _mm256_i32gather_ps(xsmall, idx, 4), acc);

        const __m256 dist_8 = _mm256_fnmadd_ps(two, acc, norm_8);

        // Batch rejection: if all 8 are >= worst, skip heap updates entirely.
        if (have_worst) {
            const int any_better = _mm256_movemask_ps(
                _mm256_cmp_ps(dist_8, _mm256_set1_ps(worst.dist), _CMP_LT_OS));
            if (any_better == 0) continue;
        }
        alignas(32) float dists[8];
        _mm256_store_ps(dists, dist_8);
        for (int v = 0; v < 8; ++v) {
            HeapPushOne(dists[v],
                        static_cast<std::int32_t>(gids[static_cast<std::size_t>(j + v)]),
                        heap, worst, have_worst);
        }
    }
    // Scalar remainder.
    for (; j < len; ++j) {
        const Code* __restrict code = codes + static_cast<std::size_t>(j) * 4u;
        const float* __restrict a = coeffs + static_cast<std::size_t>(j) * 5u;
        float acc = a[0] * root_score;
        acc += a[1] * xsmall[off1 + static_cast<int>(code[0])];
        acc += a[2] * xsmall[off2 + static_cast<int>(code[1])];
        acc += a[3] * xsmall[off3 + static_cast<int>(code[2])];
        acc += a[4] * xsmall[off4 + static_cast<int>(code[3])];
        const float dist = norm2[static_cast<std::size_t>(j)] - 2.0f * acc;
        HeapPushOne(dist,
                    static_cast<std::int32_t>(gids[static_cast<std::size_t>(j)]),
                    heap, worst, have_worst);
    }
}

STLQ_AVX2_FMA_TARGET
static void AdcScanClusterM10_Avx2(
        const Code* __restrict codes,     // stride 9
        const float* __restrict coeffs,   // stride 10
        const float* __restrict norm2,
        const std::uint32_t* __restrict gids,
        const float* __restrict xsmall,
        float root_score,
        const int* __restrict small_offsets,   // length 10
        std::uint32_t len,
        TopKHeap& heap, Candidate& worst, bool& have_worst) {
    const __m256i stride10 = _mm256_setr_epi32(0, 10, 20, 30, 40, 50, 60, 70);
    __m256i cidx[10];
    for (int l = 0; l < 10; ++l) cidx[l] = _mm256_add_epi32(stride10, _mm256_set1_epi32(l));
    __m256i tbl_off[10];
    for (int l = 1; l < 10; ++l) tbl_off[l] = _mm256_set1_epi32(small_offsets[l]);
    const __m256 two = _mm256_set1_ps(2.0f);
    const __m256 root_v = _mm256_set1_ps(root_score);

    std::uint32_t j = 0;
    for (; j + 8 <= len; j += 8) {
        const __m256 norm_8 = _mm256_loadu_ps(norm2 + j);
        const float* cb = coeffs + static_cast<std::size_t>(j) * 10u;

        // a[0] * root_score
        __m256 acc = _mm256_mul_ps(_mm256_i32gather_ps(cb, cidx[0], 4), root_v);

        // Layers 1..9: code stride is 9 bytes, not power-of-2.
        // Extract 8 code bytes per layer via scalar loads + cvtepu8_epi32.
        for (int l = 1; l < 10; ++l) {
            alignas(8) uint8_t code_buf[8];
            for (int v = 0; v < 8; ++v) {
                code_buf[v] = codes[static_cast<std::size_t>(j + v) * 9u +
                                    static_cast<std::size_t>(l - 1)];
            }
            __m256i idx = _mm256_cvtepu8_epi32(
                _mm_loadl_epi64(reinterpret_cast<const __m128i*>(code_buf)));
            idx = _mm256_add_epi32(idx, tbl_off[l]);
            const __m256 vals = _mm256_i32gather_ps(xsmall, idx, 4);
            const __m256 coeff = _mm256_i32gather_ps(cb, cidx[l], 4);
            acc = _mm256_fmadd_ps(coeff, vals, acc);
        }

        const __m256 dist_8 = _mm256_fnmadd_ps(two, acc, norm_8);

        if (have_worst) {
            const int any_better = _mm256_movemask_ps(
                _mm256_cmp_ps(dist_8, _mm256_set1_ps(worst.dist), _CMP_LT_OS));
            if (any_better == 0) continue;
        }
        alignas(32) float dists[8];
        _mm256_store_ps(dists, dist_8);
        for (int v = 0; v < 8; ++v) {
            HeapPushOne(dists[v],
                        static_cast<std::int32_t>(gids[static_cast<std::size_t>(j + v)]),
                        heap, worst, have_worst);
        }
    }
    // Scalar remainder.
    for (; j < len; ++j) {
        const Code* __restrict code = codes + static_cast<std::size_t>(j) * 9u;
        const float* __restrict a = coeffs + static_cast<std::size_t>(j) * 10u;
        float acc = a[0] * root_score;
        for (int l = 1; l < 10; ++l) {
            acc += a[l] * xsmall[small_offsets[l] + static_cast<int>(code[l - 1])];
        }
        const float dist = norm2[static_cast<std::size_t>(j)] - 2.0f * acc;
        HeapPushOne(dist,
                    static_cast<std::int32_t>(gids[static_cast<std::size_t>(j)]),
                    heap, worst, have_worst);
    }
}

#endif  // STLQ_X86_SIMD

STLQ_ALWAYS_INLINE void AdcScanClusterM5_LutSimd(
        const Code* __restrict codes,
        const float* __restrict coeffs,
        const std::uint8_t* __restrict norm2_codes,
        const float* __restrict norm2_centers,
        const std::uint32_t* __restrict gids,
        const float* __restrict xsmall,
        float root_score,
        int off1, int off2, int off3, int off4,
        std::uint32_t len,
        TopKHeap& heap, Candidate& worst, bool& have_worst) {
    for (std::uint32_t j = 0; j < len; ++j) {
        const Code* __restrict code = codes + static_cast<std::size_t>(j) * 4u;
        const float* __restrict a = coeffs + static_cast<std::size_t>(j) * 5u;
        float acc = a[0] * root_score;
        acc += a[1] * xsmall[off1 + static_cast<int>(code[0])];
        acc += a[2] * xsmall[off2 + static_cast<int>(code[1])];
        acc += a[3] * xsmall[off3 + static_cast<int>(code[2])];
        acc += a[4] * xsmall[off4 + static_cast<int>(code[3])];
        const float dist =
            norm2_centers[static_cast<int>(norm2_codes[static_cast<std::size_t>(j)])] - 2.0f * acc;
        HeapPushOne(dist,
                    static_cast<std::int32_t>(gids[static_cast<std::size_t>(j)]),
                    heap, worst, have_worst);
    }
}

STLQ_ALWAYS_INLINE void AdcScanClusterM10_LutSimd(
        const Code* __restrict codes,
        const float* __restrict coeffs,
        const std::uint8_t* __restrict norm2_codes,
        const float* __restrict norm2_centers,
        const std::uint32_t* __restrict gids,
        const float* __restrict xsmall,
        float root_score,
        const int* __restrict small_offsets,
        std::uint32_t len,
        TopKHeap& heap, Candidate& worst, bool& have_worst) {
    for (std::uint32_t j = 0; j < len; ++j) {
        const Code* __restrict code = codes + static_cast<std::size_t>(j) * 9u;
        const float* __restrict a = coeffs + static_cast<std::size_t>(j) * 10u;
        float acc = a[0] * root_score;
        #pragma omp simd reduction(+:acc)
        for (int l = 1; l < 10; ++l) {
            acc += a[l] * xsmall[small_offsets[l] + static_cast<int>(code[l - 1])];
        }
        const float dist =
            norm2_centers[static_cast<int>(norm2_codes[static_cast<std::size_t>(j)])] - 2.0f * acc;
        HeapPushOne(dist,
                    static_cast<std::int32_t>(gids[static_cast<std::size_t>(j)]),
                    heap, worst, have_worst);
    }
}

STLQ_ALWAYS_INLINE void AdcScanClusterGeneric_LutSimd(
        const Code* __restrict codes,
        const float* __restrict coeffs,
        const std::uint8_t* __restrict norm2_codes,
        const float* __restrict norm2_centers,
        const std::uint32_t* __restrict gids,
        const float* __restrict xsmall,
        float root_score,
        const int* __restrict small_offsets,
        int m,
        int m_codes,
        std::uint32_t len,
        TopKHeap& heap, Candidate& worst, bool& have_worst) {
    for (std::uint32_t j = 0; j < len; ++j) {
        const Code* __restrict code =
            codes + static_cast<std::size_t>(j) * static_cast<std::size_t>(m_codes);
        const float* __restrict a =
            coeffs + static_cast<std::size_t>(j) * static_cast<std::size_t>(m);
        float acc = a[0] * root_score;
        #pragma omp simd reduction(+:acc)
        for (int l = 1; l < m; ++l) {
            acc += a[l] * xsmall[small_offsets[l] + static_cast<int>(code[l - 1])];
        }
        const float dist =
            norm2_centers[static_cast<int>(norm2_codes[static_cast<std::size_t>(j)])] - 2.0f * acc;
        HeapPushOne(dist,
                    static_cast<std::int32_t>(gids[static_cast<std::size_t>(j)]),
                    heap, worst, have_worst);
    }
}

inline float AdcDistanceLargeRoot(const PrecompLargeRoot& pre,
                                 int cid,
                                 const Code* __restrict code_small,
                                 const float* __restrict a,
                                 const float* __restrict xCq_root,
                                 const float* __restrict xCq_small) {
    const int m = pre.m;
    const float a0 = a[0];

    // Precompute flat indices for layers 1..m-1.
    constexpr int kMaxM = 64;
    int flat_stack[kMaxM];
    int* __restrict flat = flat_stack;
#ifndef NDEBUG
    if (m > kMaxM) {
        throw std::runtime_error("AdcDistanceLargeRoot: m exceeds kMaxM.");
    }
#endif
    flat[0] = -1;
    for (int l = 1; l < m; ++l) {
        flat[l] = pre.small_offsets[l] + static_cast<int>(code_small[l - 1]);
    }

    float acc = a0 * xCq_root[cid];
    for (int l = 1; l < m; ++l) {
        acc += a[l] * xCq_small[flat[l]];
    }

    float cross0 = 0.0f;
    for (int l = 1; l < m; ++l) {
        cross0 += a[l] * GAt(pre.G0S, cid, flat[l]);
    }
    float dbnorm = a0 * a0 * pre.norm0[static_cast<std::size_t>(cid)] + 2.0f * a0 * cross0;

    float small = 0.0f;
    for (int j = 1; j < m; ++j) {
        const float aj = a[j];
        const int fj = flat[j];
        small += (aj * aj) * GAt(pre.G_small, fj, fj);
        for (int k = j + 1; k < m; ++k) {
            const float prod = 2.0f * aj * a[k];
            small += prod * GAt(pre.G_small, fj, flat[k]);
        }
    }
    dbnorm += small;

    return dbnorm - 2.0f * acc;
}

struct BaseClusterView {
    int cid = -1;
    std::uint64_t begin = 0;
    std::uint32_t len = 0;
    std::vector<std::uint32_t> gids;  // length len
    std::vector<Code> codes;          // length len*m_codes
    std::vector<float> coeffs;        // length len*m
    std::vector<float> norm2;         // length len (query-invariant term: dbnorm+small)
    eval::Norm2Lut norm2_lut;
};

STLQ_ALWAYS_INLINE float BaseNorm2LargeRoot(const PrecompLargeRoot& pre,
                                                              int cid,
                                                              const Code* __restrict code_small,
                                                              const float* __restrict a) {
    const int m = pre.m;
    const float a0 = a[0];

    constexpr int kMaxM = 64;
    int flat_stack[kMaxM];
    int* __restrict flat = flat_stack;
#ifndef NDEBUG
    if (m > kMaxM) {
        throw std::runtime_error("BaseNorm2LargeRoot: m exceeds kMaxM.");
    }
#endif
    flat[0] = -1;
    for (int l = 1; l < m; ++l) {
        flat[l] = pre.small_offsets[l] + static_cast<int>(code_small[l - 1]);
    }

    float cross0 = 0.0f;
    for (int l = 1; l < m; ++l) {
        cross0 += a[l] * GAt(pre.G0S, cid, flat[l]);
    }
    float acc_norm = (a0 * a0) * pre.norm0[static_cast<std::size_t>(cid)] + 2.0f * a0 * cross0;

    float small = 0.0f;
    for (int j = 1; j < m; ++j) {
        const float aj = a[j];
        const int fj = flat[j];
        small += (aj * aj) * GAt(pre.G_small, fj, fj);
        for (int k = j + 1; k < m; ++k) {
            const float prod = 2.0f * aj * a[k];
            small += prod * GAt(pre.G_small, fj, flat[k]);
        }
    }
    return acc_norm + small;
}

[[maybe_unused]] STLQ_ALWAYS_INLINE float BaseAccLargeRoot(const PrecompLargeRoot& pre,
                                                                             int cid,
                                                                             const Code* __restrict code_small,
                                                                             const float* __restrict a,
                                                                             const float* __restrict xCq_root,
                                                                             const float* __restrict xCq_small) {
    const int m = pre.m;
    const float root = xCq_root[cid];
    float acc = a[0] * root;
    for (int l = 1; l < m; ++l) {
        const int flat = pre.small_offsets[l] + static_cast<int>(code_small[l - 1]);
        acc += a[l] * xCq_small[flat];
    }
    return acc;
}

[[maybe_unused]] void ScanListsTopKLargeRoot(const PrecompLargeRoot& pre,
                            const io::IvfListsReader& lists,
                            io::IvfListsThreadReader* lists_reader,
                            io::BaseListThreadReader* base_list,
                            const float* __restrict xCq_root,
                            const float* __restrict xCq_small,
                            int nprobe_cap,
                            int k,
                            std::vector<int>* best_ids,
                            std::vector<float>* best_scores,
                            std::vector<std::uint32_t>* gids,
                            std::vector<Code>* codes,
                            std::vector<float>* coeffs,
                            std::vector<std::pair<float, int32_t>>* scratch,
                            float* __restrict out_dists,
                            int* __restrict out_ids) {
    const int nlist = lists.nlist();
    const int m_codes = std::max(0, pre.m - 1);
#ifndef NDEBUG
    if (!lists_reader || !base_list || !best_ids || !best_scores || !gids || !codes || !coeffs || !scratch) {
        throw std::runtime_error("ScanListsTopKLargeRoot: unexpected null pointer arguments.");
    }
    if (static_cast<int>(best_ids->size()) < nprobe_cap || static_cast<int>(best_scores->size()) < nprobe_cap) {
        throw std::runtime_error("ScanListsTopKLargeRoot: best_ids/best_scores too small for nprobe_cap.");
    }
#endif
    std::fill(best_scores->begin(), best_scores->begin() + nprobe_cap,
              -std::numeric_limits<float>::infinity());

    const int nprobe = ivf::SelectTopClustersByScore(xCq_root, nlist, nprobe_cap,
                                                     best_ids->data(), best_scores->data());

    std::vector<std::pair<float, int32_t>>& heap = *scratch;
    heap.clear();
    heap.reserve(static_cast<std::size_t>(k));

    for (int pi = 0; pi < nprobe; ++pi) {
        const int cid = (*best_ids)[static_cast<std::size_t>(pi)];
        const std::uint64_t begin = lists.Offset(cid);
        const std::uint32_t len = lists.ListSize(cid);
        if (len == 0) {
            continue;
        }

        std::string read_err;
        if (!lists_reader->ReadList(lists, cid, gids, &read_err)) {
            LogWarn("ScanListsTopKLargeRoot[cid=" + std::to_string(cid) + "]: ReadList failed: " + read_err);
            continue;
        }
        if (!base_list->ReadCodesSpan(begin, len, codes, &read_err)) {
            LogWarn("ScanListsTopKLargeRoot[cid=" + std::to_string(cid) + "]: ReadCodesSpan failed: " + read_err);
            continue;
        }
        if (!base_list->ReadCoeffsSpan(begin, len, coeffs, &read_err)) {
            LogWarn("ScanListsTopKLargeRoot[cid=" + std::to_string(cid) + "]: ReadCoeffsSpan failed: " + read_err);
            continue;
        }

        for (std::uint32_t j = 0; j < len; ++j) {
            const Code* code = codes->data() + static_cast<std::size_t>(j) * m_codes;
            const float* a = coeffs->data() + static_cast<std::size_t>(j) * pre.m;
            const float dist = AdcDistanceLargeRoot(pre, cid, code, a, xCq_root, xCq_small);
            const auto gid = static_cast<int32_t>((*gids)[static_cast<std::size_t>(j)]);

            if (static_cast<int>(heap.size()) < k) {
                heap.emplace_back(dist, gid);
                if (static_cast<int>(heap.size()) == k) {
                    std::make_heap(heap.begin(), heap.end(), ByDistThenIdLess{});
                }
                continue;
            }
            if (dist > heap.front().first ||
                (dist == heap.front().first && gid >= heap.front().second)) {
                continue;
            }
            std::pop_heap(heap.begin(), heap.end(), ByDistThenIdLess{});
            heap.back() = {dist, gid};
            std::push_heap(heap.begin(), heap.end(), ByDistThenIdLess{});
        }
    }

    if (!heap.empty()) {
        std::sort_heap(heap.begin(), heap.end(), ByDistThenIdLess{});
    }
    const int got = static_cast<int>(heap.size());
    for (int i = 0; i < got; ++i) {
        out_dists[i] = heap[static_cast<std::size_t>(i)].first;
        out_ids[i] = static_cast<int>(heap[static_cast<std::size_t>(i)].second);
    }
    for (int i = got; i < k; ++i) {
        out_dists[i] = std::numeric_limits<float>::infinity();
        out_ids[i] = -1;
    }
}

}  // namespace

template <bool ProfileTiming>
static bool EvaluateRecallBaseIvfFromDiskImpl(const Config& cfg,
                                              const Dataset& query_dataset_inmem,
                                              const io::IvfListsReader& lists,
                                              const io::BaseListReader& base_list,
                                              const TrainResult& train,
                                              RecallResult* out,
                                              DiskIvfEvalTiming* timing,
                                              std::string* err) {
    // qt_rotate_wall_sec is an INPUT set by caller via timing->qt_rotate_wall_sec.
    const double qt_rotate_wall_sec = timing ? timing->qt_rotate_wall_sec : 0.0;
    const int nth = std::max(1, (cfg.runtime.omp_threads > 0) ? cfg.runtime.omp_threads : omp_get_max_threads());
    if (cfg.runtime.omp_threads > 0) {
        omp_set_num_threads(nth);
    }
    if constexpr (ProfileTiming) {
        LogInfo("Base recall OMP: cfg.runtime.omp_threads=" + std::to_string(cfg.runtime.omp_threads) +
                " nth=" + std::to_string(nth) + " omp_max_threads=" + std::to_string(omp_get_max_threads()));
    }

    if (!out) {
        return false;
    }
    if (query_dataset_inmem.gt.size() != static_cast<std::size_t>(query_dataset_inmem.Xq.cols)) {
        if (err) *err = "EvaluateRecallBaseIvfFromDisk: invalid ground truth.";
        return false;
    }
    if (!base_list.meta().ntotal || base_list.meta().ntotal != lists.ntotal()) {
        if (err) *err = "EvaluateRecallBaseIvfFromDisk: base_list ntotal does not match ivf ntotal.";
        return false;
    }

    PrecompLargeRoot pre;
    PrecompLargeRootBuildOptions pre_opts;
    pre_opts.build_g0s_transpose = cfg.runtime.precomp_large_root_g0s_transpose;
    pre_opts.g0s_transpose_max_mb = cfg.runtime.precomp_large_root_g0s_transpose_max_mb;
    if (!BuildPrecompLargeRoot(train.C_root, /*kernels=*/nullptr, pre_opts, &pre, err)) {
        return false;
    }
    if (std::max(0, pre.m - 1) != base_list.meta().m_codes || pre.m != base_list.meta().m) {
        if (err) *err = "EvaluateRecallBaseIvfFromDisk: encoding meta does not match codebooks.";
        return false;
    }

    const int nquery = query_dataset_inmem.Xq.cols;
    const int k = std::max(1, std::min(cfg.dataset.k, static_cast<int>(lists.ntotal())));
    const int nprobe_cap = std::min(std::max(1, cfg.eval.base_nprobe), lists.nlist());
    const int qblk = 256;
    const bool use_norm2_lut = eval::IsDiskNorm2ModeLut(cfg.eval.disk_norm2_mode);

    std::vector<float> root_norm2(static_cast<std::size_t>(lists.nlist()), 1.0f);
    std::vector<float> root_inv_norm(static_cast<std::size_t>(lists.nlist()), 1.0f);
    for (int cid = 0; cid < lists.nlist(); ++cid) {
        const float* c = pre.C0.Col(cid);
        double ss = 0.0;
        for (int r = 0; r < pre.d; ++r) {
            const auto v = static_cast<double>(c[r]);
            ss += v * v;
        }
        const float norm2 = static_cast<float>(ss);
        root_norm2[static_cast<std::size_t>(cid)] = norm2;
        root_inv_norm[static_cast<std::size_t>(cid)] =
            1.0f / std::sqrt(std::max(norm2, 1e-20f));
    }

    out->indices = ColMajorMatrix<int>(k, nquery);
    out->dists = ColMajorMatrix<float>(k, nquery);

    double total_gemm_root = 0.0;
    double total_gemm_small = 0.0;
    Timer wall_timer;
    [[maybe_unused]] double total_load_sec = 0.0;
    [[maybe_unused]] double total_norm2_sec = 0.0;
    [[maybe_unused]] double total_scan_kernel_sec = 0.0;
    [[maybe_unused]] double total_topk_finalize_sec = 0.0;
    double total_probe_sel_wall_sec = 0.0;
    double total_scan_topk_wall_sec = 0.0;

    for (int q0 = 0; q0 < nquery; q0 += qblk) {
        const int qlen = std::min(qblk, nquery - q0);
        ColMajorMatrix<float> Xq_blk(pre.d, qlen);
        std::memcpy(Xq_blk.data.data(),
                    query_dataset_inmem.Xq.data.data() + static_cast<std::size_t>(q0) * static_cast<std::size_t>(pre.d),
                    sizeof(float) * static_cast<std::size_t>(pre.d) * static_cast<std::size_t>(qlen));

        ColMajorMatrix<float> xCq_root(pre.h_vec[0], qlen);
        ColMajorMatrix<float> xCq_small(pre.H_small, qlen);
        {
            ScopedBlasThreads blas_scope(OmpMaxThreads());
            Timer t0;
            Gemm(true, false, 1.0f, pre.C0, Xq_blk, 0.0f, &xCq_root);
            total_gemm_root += t0.ElapsedSeconds();
            Timer t1;
            Gemm(true, false, 1.0f, pre.C_small, Xq_blk, 0.0f, &xCq_small);
            total_gemm_small += t1.ElapsedSeconds();
        }
        std::atomic<bool> ok{true};
        std::string first_err;
        std::mutex err_mu;

        std::vector<int> q_cids_flat(static_cast<std::size_t>(qlen) * static_cast<std::size_t>(nprobe_cap), -1);
        std::vector<int> q_cids_len(static_cast<std::size_t>(qlen), 0);
        std::vector<std::uint8_t> active_flag(static_cast<std::size_t>(lists.nlist()), 0);
        std::vector<int> active_cids;
        active_cids.reserve(static_cast<std::size_t>(std::min(lists.nlist(), qlen * nprobe_cap)));

        std::vector<int> best_ids(static_cast<std::size_t>(nprobe_cap), 0);
        std::vector<float> best_scores(static_cast<std::size_t>(nprobe_cap),
                                       -std::numeric_limits<float>::infinity());
        std::vector<float> probe_scores(static_cast<std::size_t>(lists.nlist()),
                                        -std::numeric_limits<float>::infinity());
        const double probe_sel_t0 = omp_get_wtime();
        for (int qi = 0; qi < qlen; ++qi) {
            const float* scores = xCq_root.Col(qi);
            for (int cid = 0; cid < lists.nlist(); ++cid) {
                probe_scores[static_cast<std::size_t>(cid)] =
                    scores[cid] * root_inv_norm[static_cast<std::size_t>(cid)];
            }
            const int nprobe = ivf::SelectTopClustersByScore(probe_scores.data(), lists.nlist(), nprobe_cap,
                                                            best_ids.data(), best_scores.data());
            q_cids_len[static_cast<std::size_t>(qi)] = nprobe;
            for (int t = 0; t < nprobe; ++t) {
                const int cid = best_ids[static_cast<std::size_t>(t)];
                q_cids_flat[static_cast<std::size_t>(qi) * static_cast<std::size_t>(nprobe_cap) +
                            static_cast<std::size_t>(t)] = cid;
                if (active_flag[static_cast<std::size_t>(cid)] == 0) {
                    active_flag[static_cast<std::size_t>(cid)] = 1;
                    active_cids.push_back(cid);
                }
            }
        }
        total_probe_sel_wall_sec += (omp_get_wtime() - probe_sel_t0);

        std::vector<int> cid_to_active(static_cast<std::size_t>(lists.nlist()), -1);
        for (std::size_t i = 0; i < active_cids.size(); ++i) {
            cid_to_active[static_cast<std::size_t>(active_cids[i])] = static_cast<int>(i);
        }

        std::vector<BaseClusterView> active_views(active_cids.size());

        // Load per-cluster payloads once for this query block.
        double scan_topk_wall_t0 = 0.0;
        #pragma omp parallel default(none) num_threads(nth) shared(base_list, lists, pre, ok, err_mu, first_err, active_cids, active_views, xCq_root, xCq_small, q_cids_len, q_cids_flat, cid_to_active, out, scan_topk_wall_t0, total_scan_topk_wall_sec, total_load_sec, total_norm2_sec, total_scan_kernel_sec, total_topk_finalize_sec, q0, g_has_avx2_fma, use_norm2_lut, cfg) firstprivate(nth, qlen, nprobe_cap, k)
        {
            [[maybe_unused]] double local_load_sec = 0.0;
            [[maybe_unused]] double local_norm2_sec = 0.0;
            [[maybe_unused]] double local_scan_kernel_sec = 0.0;
            [[maybe_unused]] double local_topk_finalize_sec = 0.0;

            io::BaseListThreadReader base_thr;
            io::IvfListsThreadReader lists_thr;
            std::string local_err;
            if (!base_thr.OpenFrom(base_list, &local_err) || !lists_thr.OpenFrom(lists, &local_err)) {
                ok.store(false);
                std::lock_guard<std::mutex> guard(err_mu);
                if (first_err.empty()) {
                    first_err = local_err.empty() ? "EvaluateRecallBaseIvfFromDisk: failed to open thread readers."
                                                  : local_err;
                }
            }

            #pragma omp for schedule(dynamic, 1)
            for (int ai = 0; ai < static_cast<int>(active_cids.size()); ++ai) {
                if (!ok.load(std::memory_order_relaxed)) continue;
                const int cid = active_cids[static_cast<std::size_t>(ai)];
                BaseClusterView& view = active_views[static_cast<std::size_t>(ai)];
                view.cid = cid;
                view.begin = lists.Offset(cid);
                view.len = lists.ListSize(cid);
                if (view.len == 0) {
                    continue;
                }

                std::string read_err;
                const double t0 = ProfileTiming ? omp_get_wtime() : 0.0;
                if (!lists_thr.ReadList(lists, cid, &view.gids, &read_err)) {
                    LogWarn("EvaluateRecallBaseIvfFromDisk[cid=" + std::to_string(cid) + "]: ReadList failed: " + read_err);
                    view.len = 0;
                    view.gids.clear();
                    continue;
                }
                if (!base_thr.ReadCodesSpan(view.begin, view.len, &view.codes, &read_err)) {
                    LogWarn("EvaluateRecallBaseIvfFromDisk[cid=" + std::to_string(cid) + "]: ReadCodesSpan failed: " + read_err);
                    view.len = 0;
                    view.gids.clear();
                    view.codes.clear();
                    continue;
                }
                if (!base_thr.ReadCoeffsSpan(view.begin, view.len, &view.coeffs, &read_err)) {
                    LogWarn("EvaluateRecallBaseIvfFromDisk[cid=" + std::to_string(cid) + "]: ReadCoeffsSpan failed: " + read_err);
                    view.len = 0;
                    view.gids.clear();
                    view.codes.clear();
                    view.coeffs.clear();
                    continue;
                }
                if constexpr (ProfileTiming) {
                    local_load_sec += (omp_get_wtime() - t0);
                }

                const int m = pre.m;
                const int m_codes = std::max(0, m - 1);
                view.norm2.resize(static_cast<std::size_t>(view.len));
                const double t1 = ProfileTiming ? omp_get_wtime() : 0.0;
                for (std::uint32_t j = 0; j < view.len; ++j) {
                    const Code* code = view.codes.data() + static_cast<std::size_t>(j) * static_cast<std::size_t>(m_codes);
                    const float* a = view.coeffs.data() + static_cast<std::size_t>(j) * static_cast<std::size_t>(m);
                    view.norm2[static_cast<std::size_t>(j)] = BaseNorm2LargeRoot(pre, cid, code, a);
                }
                if (use_norm2_lut) {
                    if (!eval::BuildNorm2LutWithMode(view.norm2.data(),
                                                     static_cast<int>(view.norm2.size()),
                                                     cfg.base.encode.hnorms,
                                                     cfg.eval.disk_norm2_lut_kmeans_niter,
                                                     cfg.eval.disk_norm2_mode,
                                                     cfg.eval.disk_norm2_lut_log_alpha,
                                                     cfg.eval.disk_norm2_lut_piecewise_p1,
                                                     cfg.eval.disk_norm2_lut_piecewise_p2,
                                                     cfg.eval.disk_norm2_lut_piecewise_count_weight,
                                                     cfg.eval.disk_norm2_lut_piecewise_range_weight,
                                                     &view.norm2_lut,
                                                     &read_err)) {
                        LogWarn("EvaluateRecallBaseIvfFromDisk[cid=" + std::to_string(cid) +
                                "]: BuildNorm2Lut failed: " + read_err);
                        view.len = 0;
                        view.gids.clear();
                        view.codes.clear();
                        view.coeffs.clear();
                        view.norm2.clear();
                        continue;
                    }
                    view.norm2.clear();
                    view.norm2.shrink_to_fit();
                }
                if constexpr (ProfileTiming) {
                    local_norm2_sec += (omp_get_wtime() - t1);
                }
            }

            // Scan per query (CPU-only) using the preloaded clusters.
            #pragma omp single
            scan_topk_wall_t0 = omp_get_wtime();
            #pragma omp for schedule(static)
            for (int qi = 0; qi < qlen; ++qi) {
                if (!ok.load(std::memory_order_relaxed)) continue;
                const double t_scan0 = ProfileTiming ? omp_get_wtime() : 0.0;
                TopKHeap heap(k);
                const float* xroot = xCq_root.Col(qi);
                const float* xsmall = xCq_small.Col(qi);
                const int nprobe = q_cids_len[static_cast<std::size_t>(qi)];
                bool have_worst = false;
                Candidate worst{};
                for (int t = 0; t < nprobe; ++t) {
                    const int cid = q_cids_flat[static_cast<std::size_t>(qi) * static_cast<std::size_t>(nprobe_cap) +
                                                static_cast<std::size_t>(t)];
                    if (cid < 0) continue;
                    const int idx = cid_to_active[static_cast<std::size_t>(cid)];
                    if (idx < 0) continue;
                    const BaseClusterView& view = active_views[static_cast<std::size_t>(idx)];
                    if (view.len == 0) continue;

                    const int m = pre.m;
                    const int m_codes = std::max(0, m - 1);
                    const float root_score = xroot[cid];
                    const Code* __restrict codes = view.codes.data();
                    const float* __restrict coeffs = view.coeffs.data();
                    const float* __restrict norm2 = view.norm2.data();
                    const std::uint8_t* __restrict norm2_lut_u8 = view.norm2_lut.codes_u8.data();
                    const float* __restrict norm2_centers = view.norm2_lut.centers.data();
                    const std::uint32_t* __restrict gids = view.gids.data();

                    const int heap_k = heap.k;
                    if (m == 5 && m_codes == 4) {
                        const int off1 = pre.small_offsets[1];
                        const int off2 = pre.small_offsets[2];
                        const int off3 = pre.small_offsets[3];
                        const int off4 = pre.small_offsets[4];
#if defined(STLQ_X86_SIMD)
                        if (!use_norm2_lut && g_has_avx2_fma) {
                            AdcScanClusterM5_Avx2(codes, coeffs, norm2, gids, xsmall,
                                                  root_score, off1, off2, off3, off4,
                                                  view.len, heap, worst, have_worst);
                        } else
#endif
                        {
                        if (use_norm2_lut) {
                            AdcScanClusterM5_LutSimd(codes, coeffs, norm2_lut_u8, norm2_centers, gids,
                                                     xsmall, root_score, off1, off2, off3, off4,
                                                     view.len, heap, worst, have_worst);
                        } else {
                            for (std::uint32_t j = 0; j < view.len; ++j) {
                                const Code* __restrict code = codes + static_cast<std::size_t>(j) * 4u;
                                const float* __restrict a = coeffs + static_cast<std::size_t>(j) * 5u;
                                float acc = a[0] * root_score;
                                acc += a[1] * xsmall[off1 + static_cast<int>(code[0])];
                                acc += a[2] * xsmall[off2 + static_cast<int>(code[1])];
                                acc += a[3] * xsmall[off3 + static_cast<int>(code[2])];
                                acc += a[4] * xsmall[off4 + static_cast<int>(code[3])];
                                const float dist = norm2[static_cast<std::size_t>(j)] - 2.0f * acc;
                                const auto gid = static_cast<std::int32_t>(gids[static_cast<std::size_t>(j)]);

                                if (static_cast<int>(heap.heap.size()) < heap_k) {
                                    heap.heap.push_back(Candidate{dist, gid});
                                    if (static_cast<int>(heap.heap.size()) == heap_k) {
                                        heap.Heapify();
                                        worst = heap.heap[0];
                                        have_worst = true;
                                    }
                                    continue;
                                }
                                if (!have_worst) {
                                    heap.Heapify();
                                    worst = heap.heap[0];
                                    have_worst = true;
                                }
                                if (dist > worst.dist || (dist == worst.dist && gid >= worst.id)) continue;
                                heap.PushAssumeBetter(dist, gid);
                                worst = heap.heap[0];
                            }
                        }
                        }
                    } else if (m == 10 && m_codes == 9) {
                        const int off1 = pre.small_offsets[1];
                        const int off2 = pre.small_offsets[2];
                        const int off3 = pre.small_offsets[3];
                        const int off4 = pre.small_offsets[4];
                        const int off5 = pre.small_offsets[5];
                        const int off6 = pre.small_offsets[6];
                        const int off7 = pre.small_offsets[7];
                        const int off8 = pre.small_offsets[8];
                        const int off9 = pre.small_offsets[9];
#if defined(STLQ_X86_SIMD)
                        if (!use_norm2_lut && g_has_avx2_fma) {
                            AdcScanClusterM10_Avx2(codes, coeffs, norm2, gids, xsmall,
                                                   root_score, pre.small_offsets.data(),
                                                   view.len, heap, worst, have_worst);
                        } else
#endif
                        {
                        if (use_norm2_lut) {
                            AdcScanClusterM10_LutSimd(codes, coeffs, norm2_lut_u8, norm2_centers, gids,
                                                      xsmall, root_score, pre.small_offsets.data(),
                                                      view.len, heap, worst, have_worst);
                        } else {
                            for (std::uint32_t j = 0; j < view.len; ++j) {
                                const Code* __restrict code = codes + static_cast<std::size_t>(j) * 9u;
                                const float* __restrict a = coeffs + static_cast<std::size_t>(j) * 10u;
                                float acc = a[0] * root_score;
                                acc += a[1] * xsmall[off1 + static_cast<int>(code[0])];
                                acc += a[2] * xsmall[off2 + static_cast<int>(code[1])];
                                acc += a[3] * xsmall[off3 + static_cast<int>(code[2])];
                                acc += a[4] * xsmall[off4 + static_cast<int>(code[3])];
                                acc += a[5] * xsmall[off5 + static_cast<int>(code[4])];
                                acc += a[6] * xsmall[off6 + static_cast<int>(code[5])];
                                acc += a[7] * xsmall[off7 + static_cast<int>(code[6])];
                                acc += a[8] * xsmall[off8 + static_cast<int>(code[7])];
                                acc += a[9] * xsmall[off9 + static_cast<int>(code[8])];
                                const float dist = norm2[static_cast<std::size_t>(j)] - 2.0f * acc;
                                const auto gid = static_cast<std::int32_t>(gids[static_cast<std::size_t>(j)]);

                                if (static_cast<int>(heap.heap.size()) < heap_k) {
                                    heap.heap.push_back(Candidate{dist, gid});
                                    if (static_cast<int>(heap.heap.size()) == heap_k) {
                                        heap.Heapify();
                                        worst = heap.heap[0];
                                        have_worst = true;
                                    }
                                    continue;
                                }
                                if (!have_worst) {
                                    heap.Heapify();
                                    worst = heap.heap[0];
                                    have_worst = true;
                                }
                                if (dist > worst.dist || (dist == worst.dist && gid >= worst.id)) continue;
                                heap.PushAssumeBetter(dist, gid);
                                worst = heap.heap[0];
                            }
                        }
                        }
                    } else {
                        if (use_norm2_lut) {
                            AdcScanClusterGeneric_LutSimd(codes, coeffs, norm2_lut_u8, norm2_centers,
                                                          gids, xsmall, root_score,
                                                          pre.small_offsets.data(), m, m_codes,
                                                          view.len, heap, worst, have_worst);
                        } else {
                            for (std::uint32_t j = 0; j < view.len; ++j) {
                                const Code* __restrict code =
                                    codes + static_cast<std::size_t>(j) * static_cast<std::size_t>(m_codes);
                                const float* __restrict a =
                                    coeffs + static_cast<std::size_t>(j) * static_cast<std::size_t>(m);
                                float acc = a[0] * root_score;
                                for (int l = 1; l < m; ++l) {
                                    const int flat = pre.small_offsets[l] + static_cast<int>(code[l - 1]);
                                    acc += a[l] * xsmall[flat];
                                }
                                const float dist = norm2[static_cast<std::size_t>(j)] - 2.0f * acc;
                                const auto gid = static_cast<std::int32_t>(gids[static_cast<std::size_t>(j)]);

                                if (static_cast<int>(heap.heap.size()) < heap_k) {
                                    heap.heap.push_back(Candidate{dist, gid});
                                    if (static_cast<int>(heap.heap.size()) == heap_k) {
                                        heap.Heapify();
                                        worst = heap.heap[0];
                                        have_worst = true;
                                    }
                                    continue;
                                }
                                if (!have_worst) {
                                    heap.Heapify();
                                    worst = heap.heap[0];
                                    have_worst = true;
                                }
                                if (dist > worst.dist || (dist == worst.dist && gid >= worst.id)) continue;
                                heap.PushAssumeBetter(dist, gid);
                                worst = heap.heap[0];
                            }
                        }
                    }
                }
                const double t_scan1 = ProfileTiming ? omp_get_wtime() : 0.0;
                if constexpr (ProfileTiming) {
                    local_scan_kernel_sec += (t_scan1 - t_scan0);
                }
                const double t_fin0 = ProfileTiming ? omp_get_wtime() : 0.0;
                heap.Finalize(out->dists.Col(q0 + qi), out->indices.Col(q0 + qi));
                if constexpr (ProfileTiming) {
                    local_topk_finalize_sec += (omp_get_wtime() - t_fin0);
                }
            }
            #pragma omp single
            total_scan_topk_wall_sec += (omp_get_wtime() - scan_topk_wall_t0);

            if constexpr (ProfileTiming) {
                #pragma omp atomic
                total_load_sec += local_load_sec;
                #pragma omp atomic
                total_norm2_sec += local_norm2_sec;
                #pragma omp atomic
                total_scan_kernel_sec += local_scan_kernel_sec;
                #pragma omp atomic
                total_topk_finalize_sec += local_topk_finalize_sec;
            }
        }
        if (!ok.load()) {
            if (err) {
                *err = first_err.empty() ? "EvaluateRecallBaseIvfFromDisk: failed." : first_err;
            }
            return false;
        }
    }

    const double wall_sec = wall_timer.ElapsedSeconds();
    if constexpr (ProfileTiming) {
        if (!cfg.eval.bench_quiet) {
            LogInfo("---- Disk IVF base timing summary (wall/core) ----");
        }
    }
    const int nq = query_dataset_inmem.Xq.cols;
    const double qt_gemm_wall_sec = total_gemm_root + total_gemm_small;
    const double core_wall_sec = qt_rotate_wall_sec + qt_gemm_wall_sec +
                                 total_probe_sel_wall_sec + total_scan_topk_wall_sec;
    if (timing) {
        timing->wall_sec = wall_sec;
        timing->core_wall_sec = core_wall_sec;
        // qt_rotate_wall_sec is INPUT (set by caller); do NOT overwrite.
        timing->qt_gemm_wall_sec = qt_gemm_wall_sec;
        timing->probe_sel_wall_sec = total_probe_sel_wall_sec;
        timing->scan_topk_wall_sec = total_scan_topk_wall_sec;
        timing->gemm_root_wall_sec = total_gemm_root;
        timing->gemm_small_wall_sec = total_gemm_small;
    }
    if (!cfg.eval.bench_quiet) {
        if (qt_rotate_wall_sec > 0.0) {
            LogInfo("Disk IVF base qt_rotate time: " + std::to_string(qt_rotate_wall_sec) + "s");
        }
        LogInfo("Disk IVF base qt_build GEMM time: " + std::to_string(qt_gemm_wall_sec) +
                "s (root=" + std::to_string(total_gemm_root) +
                "s small=" + std::to_string(total_gemm_small) + "s)");
        LogInfo("Disk IVF base probe_sel time: " + std::to_string(total_probe_sel_wall_sec) + "s");
        LogInfo("Disk IVF base scan+topk time: " + std::to_string(total_scan_topk_wall_sec) + "s");
        LogInfo("Disk IVF base wall time (qt_rotate+qt_gemm+probe_sel+scan+topk): " +
                std::to_string(core_wall_sec) + "s");
        if (nq > 0 && core_wall_sec > 0.0) {
            LogInfo("Disk IVF base QPS (wall): " + std::to_string(static_cast<double>(nq) / core_wall_sec));
        }
    }
    if constexpr (ProfileTiming) {
        if (!cfg.eval.bench_quiet) {
            LogInfo("---- Disk IVF base timing breakdown (profile; sum over threads) ----");
            LogInfo("Disk IVF base load clusters time (sum over threads): " + std::to_string(total_load_sec) + "s");
            LogInfo("Disk IVF base norm2 precompute time (sum over threads): " + std::to_string(total_norm2_sec) + "s");
            LogInfo("Disk IVF base scan kernel time (sum over threads): " + std::to_string(total_scan_kernel_sec) + "s");
            LogInfo("Disk IVF base topk finalize time (sum over threads): " + std::to_string(total_topk_finalize_sec) + "s");
            LogInfo("Disk IVF base scan+topk time (sum over threads): " +
                    std::to_string(total_scan_kernel_sec + total_topk_finalize_sec) + "s");
                LogInfo("Disk IVF base core time (accounted, qt_rotate+qt_gemm+probe_sel+scan+topk): " +
                    std::to_string(qt_rotate_wall_sec + qt_gemm_wall_sec + total_probe_sel_wall_sec +
                           total_scan_kernel_sec + total_topk_finalize_sec) + "s");
        }
    }

    out->recall = ComputeRecallAtKMatrix(query_dataset_inmem.gt, out->indices);
    return true;
}

bool EvaluateRecallBaseIvfFromDisk(const Config& cfg,
                                   const Dataset& query_dataset_inmem,
                                   const io::IvfListsReader& lists,
                                   const io::BaseListReader& base_list,
                                   const TrainResult& train,
                                   RecallResult* out,
                                   std::string* err) {
    DiskIvfEvalTiming local_t{};
    local_t.qt_rotate_wall_sec = 0.0;
    return EvaluateRecallBaseIvfFromDiskTimed(cfg, query_dataset_inmem, lists, base_list, train,
                                              out, &local_t, err);
}

bool EvaluateRecallBaseIvfFromDiskTimed(const Config& cfg,
                                        const Dataset& query_dataset_inmem,
                                        const io::IvfListsReader& lists,
                                        const io::BaseListReader& base_list,
                                        const TrainResult& train,
                                        RecallResult* out,
                                        DiskIvfEvalTiming* timing,
                                        std::string* err) {
    // qt_rotate_wall_sec is read from timing->qt_rotate_wall_sec (caller sets it).
    if (cfg.large.profile_timing) {
        return EvaluateRecallBaseIvfFromDiskImpl<true>(cfg, query_dataset_inmem, lists, base_list,
                                                       train, out, timing, err);
    }
    return EvaluateRecallBaseIvfFromDiskImpl<false>(cfg, query_dataset_inmem, lists, base_list,
                                                    train, out, timing, err);
}

}  // namespace stlq
