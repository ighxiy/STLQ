#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include <omp.h>

namespace stlq::ivf {

namespace detail {

inline void SiftUpMinHeap(float* __restrict heap_scores,
                          int* __restrict heap_ids,
                          int idx) {
    // "hole" sift-up: fewer moves than swap-per-level.
    float s = heap_scores[idx];
    int id = heap_ids[idx];
    while (idx > 0) {
        const int parent = (idx - 1) >> 1;
        const float ps = heap_scores[parent];
        if (ps <= s) {
            break;
        }
        heap_scores[idx] = ps;
        heap_ids[idx] = heap_ids[parent];
        idx = parent;
    }
    heap_scores[idx] = s;
    heap_ids[idx] = id;
}

inline void SiftDownMinHeap(float* __restrict heap_scores,
                            int* __restrict heap_ids,
                            int size,
                            int idx) {
    // "hole" sift-down: fewer moves than swap-per-level.
    float s = heap_scores[idx];
    int id = heap_ids[idx];
    while (true) {
        const int left = (idx << 1) + 1;
        if (left >= size) {
            break;
        }
        const int right = left + 1;
        int child = left;
        float cs = heap_scores[left];
        if (right < size) {
            const float rs = heap_scores[right];
            if (rs < cs) {
                child = right;
                cs = rs;
            }
        }
        if (s <= cs) {
            break;
        }
        heap_scores[idx] = cs;
        heap_ids[idx] = heap_ids[child];
        idx = child;
    }
    heap_scores[idx] = s;
    heap_ids[idx] = id;
}

}  // namespace detail

// Selects top-`nprobe` cluster ids by score (larger is better) from `scores[0..nlist)`.
// Uses a fixed-size min-heap: O(nlist log nprobe) time, O(nprobe) extra storage.
//
// Output:
// - `out_ids[0..nprobe)` holds selected cluster ids (order unspecified).
// - `heap_scores[0..nprobe)` is a scratch buffer used as heap keys.
// Returns the number of selected clusters (== min(nprobe, nlist)).
inline int SelectTopClustersByScore(const float* __restrict scores,
                                   int nlist,
                                   int nprobe,
                                   int* __restrict out_ids,
                                   float* __restrict heap_scores) {
    if (nlist <= 0 || nprobe <= 0) {
        return 0;
    }
    nprobe = std::min(nprobe, nlist);

    int heap_size = 0;
    for (int cid = 0; cid < nlist; ++cid) {
        const float s = scores[cid];
        if (heap_size < nprobe) {
            heap_scores[heap_size] = s;
            out_ids[heap_size] = cid;
            detail::SiftUpMinHeap(heap_scores, out_ids, heap_size);
            ++heap_size;
            continue;
        }
        if (s <= heap_scores[0]) {
            continue;
        }
        heap_scores[0] = s;
        out_ids[0] = cid;
        detail::SiftDownMinHeap(heap_scores, out_ids, heap_size, 0);
    }
    return heap_size;
}

// ---------- Two-level hierarchical (hier2) IVF probe selection ----------

struct Hier2Split {
    int K = 0;   // original flat cluster count
    int K1 = 0;  // number of coarse groups (spherical k-means clusters)
    // Per-coarse-group membership: groups[c] = sorted list of fine centroid IDs assigned to coarse group c.
    std::vector<std::vector<int>> groups;
};

/// Compute default K1 (number of coarse groups) for a given K.
/// K1 ~ sqrt(K), rounded to nearest power-of-two in [128, 1024].
inline int ComputeHier2DefaultK1(int K) {
    if (K <= 0) return 0;
    int K1 = static_cast<int>(std::lround(std::sqrt(static_cast<double>(K))));
    K1 = std::max(128, std::min(1024, K1));
    // Round to nearest power of two.
    {
        int down = 1;
        while ((down << 1) > 0 && (down << 1) <= K1) down <<= 1;
        int up = 1;
        while (up > 0 && up < K1) up <<= 1;
        K1 = (std::abs(up - K1) <= std::abs(K1 - down)) ? up : down;
    }
    K1 = std::max(128, std::min(1024, K1));
    if (K1 > K) K1 = K;
    return K1;
}

/// Build a Hier2Split by running spherical k-means on the fine centroids.
///
/// @param fine    Column-major matrix (d x K) of fine centroids (assumed L2-normalized for spherical IVF).
/// @param K       Number of fine centroids (== fine.cols).
/// @param K1      Desired number of coarse groups.
/// @param kmeans_iters  Number of k-means iterations (10-20 is usually enough).
/// @param seed    Random seed for k-means initialization.
/// @return        Hier2Split with K, K1, and groups[] populated.
///
/// The resulting coarse centroids are NOT stored in Hier2Split; the caller builds them
/// separately from the assignment (via BuildCoarseFromHier2).
inline Hier2Split BuildHier2SplitKmeans(const float* fine_data, int d, int K, int K1,
                                        int kmeans_iters = 15, unsigned seed = 12345u) {
    Hier2Split s;
    s.K = K;
    s.K1 = std::max(1, std::min(K1, K));
    s.groups.resize(static_cast<std::size_t>(s.K1));

    if (K <= 0 || d <= 0) return s;

    // --- Spherical k-means ---
    // 1) Initialize coarse centroids: pick K1 fine centroids spread evenly (deterministic).
    std::vector<float> centroids(static_cast<std::size_t>(s.K1) * d);
    {
        // Spread-init: pick indices at regular intervals, then shuffle with seed for variety.
        std::vector<int> init_ids(static_cast<std::size_t>(s.K1));
        for (int c = 0; c < s.K1; ++c) {
            init_ids[static_cast<std::size_t>(c)] =
                static_cast<int>(static_cast<std::int64_t>(c) * K / s.K1);
        }
        // Simple LCG shuffle seeded by `seed` — avoids <random> header dependency.
        {
            unsigned rng = seed;
            for (int i = s.K1 - 1; i > 0; --i) {
                rng = rng * 1664525u + 1013904223u;
                const int j = static_cast<int>(rng % static_cast<unsigned>(i + 1));
                std::swap(init_ids[static_cast<std::size_t>(i)],
                          init_ids[static_cast<std::size_t>(j)]);
            }
        }
        for (int c = 0; c < s.K1; ++c) {
            const int fid = init_ids[static_cast<std::size_t>(c)];
            const float* src = fine_data + static_cast<std::size_t>(fid) * d;
            float* dst = centroids.data() + static_cast<std::size_t>(c) * d;
            for (int r = 0; r < d; ++r) dst[r] = src[r];
        }
    }

    // 2) Iterate: assign each fine centroid to nearest coarse centroid (by dot product),
    //    then recompute coarse centroids as normalized mean of assigned fine centroids.
    std::vector<int> assign(static_cast<std::size_t>(K), 0);
    const int nt = std::max(1, omp_get_max_threads());
    std::vector<float> partial(static_cast<std::size_t>(nt) *
                               static_cast<std::size_t>(s.K1) *
                               static_cast<std::size_t>(d),
                               0.0f);
    for (int iter = 0; iter < kmeans_iters; ++iter) {
        // --- Assignment step ---
        #pragma omp parallel for schedule(static)
        for (int fid = 0; fid < K; ++fid) {
            const float* fv = fine_data + static_cast<std::size_t>(fid) * d;
            float best_dot = -std::numeric_limits<float>::infinity();
            int best_c = 0;
            for (int c = 0; c < s.K1; ++c) {
                const float* cv = centroids.data() + static_cast<std::size_t>(c) * d;
                float dot = 0.0f;
                #pragma omp simd reduction(+:dot)
                for (int r = 0; r < d; ++r) dot += fv[r] * cv[r];
                if (dot > best_dot) { best_dot = dot; best_c = c; }
            }
            assign[static_cast<std::size_t>(fid)] = best_c;
        }

        // --- Update step: recompute coarse centroids as L2-normalized mean ---
        std::fill(centroids.begin(), centroids.end(), 0.0f);
        std::fill(partial.begin(), partial.end(), 0.0f);
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            const std::size_t base = static_cast<std::size_t>(tid) *
                                     static_cast<std::size_t>(s.K1) *
                                     static_cast<std::size_t>(d);
            float* local = partial.data() + base;
            #pragma omp for schedule(static)
            for (int fid = 0; fid < K; ++fid) {
                const int c = assign[static_cast<std::size_t>(fid)];
                const float* fv = fine_data + static_cast<std::size_t>(fid) * d;
                float* cv = local + static_cast<std::size_t>(c) * d;
                for (int r = 0; r < d; ++r) cv[r] += fv[r];
            }
        }
        for (int tid = 0; tid < nt; ++tid) {
            const std::size_t base = static_cast<std::size_t>(tid) *
                                     static_cast<std::size_t>(s.K1) *
                                     static_cast<std::size_t>(d);
            const float* local = partial.data() + base;
            for (int c = 0; c < s.K1; ++c) {
                float* cv = centroids.data() + static_cast<std::size_t>(c) * d;
                const float* lv = local + static_cast<std::size_t>(c) * d;
                for (int r = 0; r < d; ++r) cv[r] += lv[r];
            }
        }

        #pragma omp parallel for schedule(static)
        for (int c = 0; c < s.K1; ++c) {
            float* cv = centroids.data() + static_cast<std::size_t>(c) * d;
            float ss = 0.0f;
            #pragma omp simd reduction(+:ss)
            for (int r = 0; r < d; ++r) ss += cv[r] * cv[r];
            if (ss > 1e-12f) {
                const float inv = 1.0f / std::sqrt(ss);
                for (int r = 0; r < d; ++r) cv[r] *= inv;
            }
        }
    }

    // 3) Build group membership lists from final assignment.
    for (auto& g : s.groups) g.clear();
    for (int fid = 0; fid < K; ++fid) {
        const int c = assign[static_cast<std::size_t>(fid)];
        s.groups[static_cast<std::size_t>(c)].push_back(fid);
    }
    // Groups are naturally sorted by fid insertion order (ascending).
    return s;
}

/// Build coarse centroids (d x K1) from a Hier2Split's group membership.
/// Each coarse centroid = L2-normalized mean of assigned fine centroids.
/// Returned matrix has shape (d, K1); column c is the coarse centroid for group c.
inline std::vector<float> BuildCoarseFromHier2(const float* fine_data, int d,
                                               const Hier2Split& split) {
    const int K1 = split.K1;
    std::vector<float> coarse(static_cast<std::size_t>(K1) * d, 0.0f);
    for (int c = 0; c < K1; ++c) {
        float* out = coarse.data() + static_cast<std::size_t>(c) * d;
        for (int fid : split.groups[static_cast<std::size_t>(c)]) {
            const float* fv = fine_data + static_cast<std::size_t>(fid) * d;
            for (int r = 0; r < d; ++r) out[r] += fv[r];
        }
        float ss = 0.0f;
        for (int r = 0; r < d; ++r) ss += out[r] * out[r];
        if (ss > 1e-12f) {
            const float inv = 1.0f / std::sqrt(ss);
            for (int r = 0; r < d; ++r) out[r] *= inv;
        }
    }
    return coarse;
}

/// Select top-L coarse groups by score (larger is better).
/// If cent_norm2 is non-null, score = 2*dot - norm2 (L2 proxy); otherwise score = dot.
/// Returns the number of selected groups (== min(topL, K1)).
/// Supports arbitrary topL up to K1: topL=1/2 use fast scalar paths; topL>2 uses partial_sort.
inline int SelectTopCoarseByScore(const float* __restrict dot,
                                  const float* __restrict cent_norm2,
                                  int K1,
                                  int topL,
                                  int* __restrict out_ids) {
    if (!dot || !out_ids || K1 <= 0) return 0;
    topL = std::max(1, std::min(topL, K1));

    if (topL == 1) {
        float best = -std::numeric_limits<float>::infinity();
        int best_id = 0;
        for (int cid = 0; cid < K1; ++cid) {
            const float score = cent_norm2 ? (2.0f * dot[cid] - cent_norm2[cid]) : dot[cid];
            if (score > best) { best = score; best_id = cid; }
        }
        out_ids[0] = best_id;
        return 1;
    }
    if (topL == 2 && K1 >= 2) {
        float a0 = -std::numeric_limits<float>::infinity();
        float a1 = -std::numeric_limits<float>::infinity();
        int ai0 = 0, ai1 = 0;
        for (int cid = 0; cid < K1; ++cid) {
            const float score = cent_norm2 ? (2.0f * dot[cid] - cent_norm2[cid]) : dot[cid];
            if (score > a0 || (score == a0 && cid < ai0)) {
                a1 = a0; ai1 = ai0; a0 = score; ai0 = cid;
            } else if (score > a1 || (score == a1 && cid < ai1)) {
                a1 = score; ai1 = cid;
            }
        }
        out_ids[0] = ai0;
        out_ids[1] = ai1;
        return 2;
    }
    // General path: topL > 2.  K1 is typically small (128-1024), so partial_sort is fine.
    static thread_local std::vector<std::pair<float, int>> scratch;
    scratch.resize(static_cast<std::size_t>(K1));
    for (int cid = 0; cid < K1; ++cid) {
        scratch[cid] = {cent_norm2 ? (2.0f * dot[cid] - cent_norm2[cid]) : dot[cid], cid};
    }
    std::partial_sort(scratch.begin(), scratch.begin() + topL, scratch.end(),
                      [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                          return a.first > b.first;
                      });
    for (int i = 0; i < topL; ++i) out_ids[i] = scratch[i].second;
    return topL;
}

}  // namespace stlq::ivf

