#include "stlq/eval/norm2_lut.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>

#include "stlq/pipeline/large_store_hash.h"
#include "stlq/io/linkage_list_store.h"
#include <omp.h>

namespace stlq::eval {

namespace {

struct Hash64 {
    std::uint64_t h = 1469598103934665603ull;

    void AddBytes(const void* data, std::size_t len) {
        const auto* p = static_cast<const std::uint8_t*>(data);
        for (std::size_t i = 0; i < len; ++i) {
            h ^= static_cast<std::uint64_t>(p[i]);
            h *= 1099511628211ull;
        }
    }
    void AddU64(std::uint64_t v) { AddBytes(&v, sizeof(v)); }
    void AddI32(std::int32_t v) { AddBytes(&v, sizeof(v)); }
    void AddStr(const std::string& s) {
        AddU64(static_cast<std::uint64_t>(s.size()));
        if (!s.empty()) AddBytes(s.data(), s.size());
    }
};

enum class Norm2LutTransform {
    kIdentity = 0,
    kSqrt = 1,
    kLog1p = 2,
};

Norm2LutTransform ResolveNorm2LutTransform(const std::string& mode) {
    if (mode == "lut_sqrt") return Norm2LutTransform::kSqrt;
    if (mode == "lut_log1p") return Norm2LutTransform::kLog1p;
    return Norm2LutTransform::kIdentity;
}

float ClampNonNegative(float x) {
    return std::max(0.0f, x);
}

float ForwardNorm2LutTransform(float x, Norm2LutTransform tfm, double log1p_alpha) {
    x = ClampNonNegative(x);
    switch (tfm) {
        case Norm2LutTransform::kSqrt:
            return std::sqrt(x);
        case Norm2LutTransform::kLog1p: {
            const double alpha = std::max(1e-12, log1p_alpha);
            return static_cast<float>(std::log1p(alpha * static_cast<double>(x)));
        }
        case Norm2LutTransform::kIdentity:
        default:
            return x;
    }
}

float InverseNorm2LutTransform(float y, Norm2LutTransform tfm, double log1p_alpha) {
    y = ClampNonNegative(y);
    switch (tfm) {
        case Norm2LutTransform::kSqrt:
            return y * y;
        case Norm2LutTransform::kLog1p: {
            const double alpha = std::max(1e-12, log1p_alpha);
            return static_cast<float>(std::max(0.0, std::expm1(static_cast<double>(y)) / alpha));
        }
        case Norm2LutTransform::kIdentity:
        default:
            return y;
    }
}

float SortedQuantileValue(const std::vector<float>& sorted, double q) {
    if (sorted.empty()) return 0.0f;
    q = std::min(1.0, std::max(0.0, q));
    const double pos = q * static_cast<double>(std::max<std::size_t>(1, sorted.size()) - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
    const std::size_t hi = std::min(sorted.size() - 1, lo + 1);
    const double frac = pos - static_cast<double>(lo);
    return static_cast<float>((1.0 - frac) * static_cast<double>(sorted[lo]) +
                              frac * static_cast<double>(sorted[hi]));
}

float TailWeightForValue(float x, float p99, float p999) {
    if (x > p999) return 16.0f;
    if (x > p99) return 4.0f;
    return 1.0f;
}

void Assign1DToCenters(const float* data,
                       int n,
                       const std::vector<float>& centers,
                       std::vector<int>* assignments) {
    assignments->assign(static_cast<std::size_t>(std::max(0, n)), 0);
    if (n <= 0 || centers.empty()) {
        return;
    }
    const int k = static_cast<int>(centers.size());
    #pragma omp parallel for schedule(static) default(none) shared(data, assignments, centers) firstprivate(n, k)
    for (int i = 0; i < n; ++i) {
        const float x = data[i];
        float best = std::fabs(x - centers[0]);
        int best_k = 0;
        for (int j = 1; j < k; ++j) {
            const float dist = std::fabs(x - centers[static_cast<std::size_t>(j)]);
            const bool closer = (dist < best);
            best = closer ? dist : best;
            best_k = closer ? j : best_k;
        }
        (*assignments)[static_cast<std::size_t>(i)] = best_k;
    }
}

void Kmeans1DQuantize(const float* data,
                      int n,
                      int k,
                      int max_iter,
                      std::vector<int>* assignments,
                      std::vector<float>* centers) {
    assignments->assign(static_cast<std::size_t>(std::max(0, n)), 0);
    centers->assign(static_cast<std::size_t>(std::max(0, k)), 0.0f);
    if (n <= 0 || k <= 0) {
        return;
    }
    if (k == 1) {
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            sum += data[i];
        }
        (*centers)[0] = static_cast<float>(sum / static_cast<double>(n));
        return;
    }

    std::vector<float> sorted(static_cast<std::size_t>(n));
    std::copy(data, data + n, sorted.begin());
    std::sort(sorted.begin(), sorted.end());
    for (int i = 0; i < k; ++i) {
        const int idx = static_cast<int>((static_cast<long long>(i) * n) / k);
        (*centers)[static_cast<std::size_t>(i)] =
            sorted[static_cast<std::size_t>(std::min(idx, n - 1))];
    }

    const int threads = std::max(1, omp_get_max_threads());
    std::vector<int> counts(static_cast<std::size_t>(threads) * static_cast<std::size_t>(k), 0);
    std::vector<double> sums(static_cast<std::size_t>(threads) * static_cast<std::size_t>(k), 0.0);

    for (int it = 0; it < max_iter; ++it) {
        std::fill(counts.begin(), counts.end(), 0);
        std::fill(sums.begin(), sums.end(), 0.0);

        #pragma omp parallel default(none) shared(data, assignments, centers, counts, sums) firstprivate(n, k)
        {
            const int tid = omp_get_thread_num();
            int* __restrict c_local = counts.data() + static_cast<std::size_t>(tid) * static_cast<std::size_t>(k);
            double* __restrict s_local = sums.data() + static_cast<std::size_t>(tid) * static_cast<std::size_t>(k);

            #pragma omp for schedule(static)
            for (int i = 0; i < n; ++i) {
                const float x = data[i];
                float best = std::fabs(x - (*centers)[0]);
                int best_k = 0;
                for (int j = 1; j < k; ++j) {
                    const float dist = std::fabs(x - (*centers)[static_cast<std::size_t>(j)]);
                    const bool closer = (dist < best);
                    best = closer ? dist : best;
                    best_k = closer ? j : best_k;
                }
                (*assignments)[static_cast<std::size_t>(i)] = best_k;
                c_local[best_k] += 1;
                s_local[best_k] += static_cast<double>(x);
            }
        }

        bool converged = true;
        for (int j = 0; j < k; ++j) {
            int cnt = 0;
            double sum = 0.0;
            for (int t = 0; t < threads; ++t) {
                cnt += counts[static_cast<std::size_t>(t) * static_cast<std::size_t>(k) +
                              static_cast<std::size_t>(j)];
                sum += sums[static_cast<std::size_t>(t) * static_cast<std::size_t>(k) +
                            static_cast<std::size_t>(j)];
            }
            if (cnt <= 0) {
                continue;
            }
            const float newc = static_cast<float>(sum / static_cast<double>(cnt));
            if (std::fabs(newc - (*centers)[static_cast<std::size_t>(j)]) > 1e-6f) {
                converged = false;
            }
            (*centers)[static_cast<std::size_t>(j)] = newc;
        }
        if (converged) {
            break;
        }
    }

    // The loop updates centers after computing assignments. Re-run one final
    // assignment pass so the returned codes correspond to the final centers.
    Assign1DToCenters(data, n, *centers, assignments);
}

void Kmeans1DQuantizeTailWeighted(const float* data,
                                  int n,
                                  int k,
                                  int max_iter,
                                  std::vector<int>* assignments,
                                  std::vector<float>* centers) {
    assignments->assign(static_cast<std::size_t>(std::max(0, n)), 0);
    centers->assign(static_cast<std::size_t>(std::max(0, k)), 0.0f);
    if (n <= 0 || k <= 0) {
        return;
    }

    std::vector<float> sorted(static_cast<std::size_t>(n));
    std::copy(data, data + n, sorted.begin());
    std::sort(sorted.begin(), sorted.end());
    const float p99 = SortedQuantileValue(sorted, 0.99);
    const float p999 = SortedQuantileValue(sorted, 0.999);

    if (k == 1) {
        double sum = 0.0;
        double total_w = 0.0;
        for (int i = 0; i < n; ++i) {
            const double w = static_cast<double>(TailWeightForValue(data[i], p99, p999));
            sum += w * static_cast<double>(data[i]);
            total_w += w;
        }
        (*centers)[0] = static_cast<float>(sum / std::max(1e-12, total_w));
        return;
    }

    double total_w = 0.0;
    for (int i = 0; i < n; ++i) {
        total_w += static_cast<double>(TailWeightForValue(sorted[static_cast<std::size_t>(i)], p99, p999));
    }
    double prefix_w = 0.0;
    int sorted_idx = 0;
    for (int j = 0; j < k; ++j) {
        const double target = (static_cast<double>(j) * total_w) / static_cast<double>(k);
        while (sorted_idx + 1 < n && prefix_w < target) {
            prefix_w += static_cast<double>(
                TailWeightForValue(sorted[static_cast<std::size_t>(sorted_idx)], p99, p999));
            ++sorted_idx;
        }
        (*centers)[static_cast<std::size_t>(j)] =
            sorted[static_cast<std::size_t>(std::min(sorted_idx, n - 1))];
    }

    const int threads = std::max(1, omp_get_max_threads());
    std::vector<double> counts(static_cast<std::size_t>(threads) * static_cast<std::size_t>(k), 0.0);
    std::vector<double> sums(static_cast<std::size_t>(threads) * static_cast<std::size_t>(k), 0.0);

    for (int it = 0; it < max_iter; ++it) {
        std::fill(counts.begin(), counts.end(), 0.0);
        std::fill(sums.begin(), sums.end(), 0.0);

        #pragma omp parallel default(none) shared(data, assignments, centers, counts, sums) firstprivate(n, k, p99, p999)
        {
            const int tid = omp_get_thread_num();
            double* __restrict c_local = counts.data() + static_cast<std::size_t>(tid) * static_cast<std::size_t>(k);
            double* __restrict s_local = sums.data() + static_cast<std::size_t>(tid) * static_cast<std::size_t>(k);

            #pragma omp for schedule(static)
            for (int i = 0; i < n; ++i) {
                const float x = data[i];
                float best = std::fabs(x - (*centers)[0]);
                int best_k = 0;
                for (int j = 1; j < k; ++j) {
                    const float dist = std::fabs(x - (*centers)[static_cast<std::size_t>(j)]);
                    const bool closer = (dist < best);
                    best = closer ? dist : best;
                    best_k = closer ? j : best_k;
                }
                (*assignments)[static_cast<std::size_t>(i)] = best_k;
                const double w = static_cast<double>(TailWeightForValue(x, p99, p999));
                c_local[best_k] += w;
                s_local[best_k] += w * static_cast<double>(x);
            }
        }

        bool converged = true;
        for (int j = 0; j < k; ++j) {
            double cnt = 0.0;
            double sum = 0.0;
            for (int t = 0; t < threads; ++t) {
                cnt += counts[static_cast<std::size_t>(t) * static_cast<std::size_t>(k) +
                              static_cast<std::size_t>(j)];
                sum += sums[static_cast<std::size_t>(t) * static_cast<std::size_t>(k) +
                            static_cast<std::size_t>(j)];
            }
            if (cnt <= 0.0) {
                continue;
            }
            const float newc = static_cast<float>(sum / cnt);
            if (std::fabs(newc - (*centers)[static_cast<std::size_t>(j)]) > 1e-6f) {
                converged = false;
            }
            (*centers)[static_cast<std::size_t>(j)] = newc;
        }
        if (converged) {
            break;
        }
    }

    Assign1DToCenters(data, n, *centers, assignments);
}

std::vector<int> AllocatePiecewiseBins(const std::vector<int>& counts,
                                       const std::vector<float>& mins,
                                       const std::vector<float>& maxs,
                                       int k,
                                       double count_weight,
                                       double range_weight) {
    const int nseg = static_cast<int>(counts.size());
    std::vector<int> bins(static_cast<std::size_t>(nseg), 0);
    if (k <= 0 || nseg <= 0) return bins;
    int nonempty = 0;
    int total_count = 0;
    float total_range = 0.0f;
    for (int s = 0; s < nseg; ++s) {
        if (counts[static_cast<std::size_t>(s)] > 0) {
            ++nonempty;
            total_count += counts[static_cast<std::size_t>(s)];
            total_range += std::max(0.0f, maxs[static_cast<std::size_t>(s)] - mins[static_cast<std::size_t>(s)]);
        }
    }
    if (nonempty <= 0) return bins;
    if (k < nonempty) {
        for (int s = 0; s < nseg && k > 0; ++s) {
            if (counts[static_cast<std::size_t>(s)] > 0) {
                bins[static_cast<std::size_t>(s)] = 1;
                --k;
            }
        }
        return bins;
    }
    for (int s = 0; s < nseg; ++s) {
        if (counts[static_cast<std::size_t>(s)] > 0) {
            bins[static_cast<std::size_t>(s)] = 1;
        }
    }
    int remaining = k - nonempty;
    while (remaining > 0) {
        int best_s = -1;
        double best_priority = -1.0;
        for (int s = 0; s < nseg; ++s) {
            const int cnt = counts[static_cast<std::size_t>(s)];
            if (cnt <= 0 || bins[static_cast<std::size_t>(s)] >= cnt) continue;
            const double count_share = static_cast<double>(cnt) / static_cast<double>(std::max(1, total_count));
            const float range = std::max(0.0f, maxs[static_cast<std::size_t>(s)] - mins[static_cast<std::size_t>(s)]);
            const double range_share =
                (total_range > 1e-12f) ? (static_cast<double>(range) / static_cast<double>(total_range)) : count_share;
            const double score = count_weight * count_share + range_weight * range_share;
            const double priority = score / static_cast<double>(bins[static_cast<std::size_t>(s)] + 1);
            if (priority > best_priority) {
                best_priority = priority;
                best_s = s;
            }
        }
        if (best_s < 0) break;
        ++bins[static_cast<std::size_t>(best_s)];
        --remaining;
    }
    return bins;
}

bool BuildNorm2LutPiecewise(const float* data,
                            int n,
                            int requested_centers,
                            double piecewise_p1,
                            double piecewise_p2,
                            double piecewise_count_weight,
                            double piecewise_range_weight,
                            Norm2Lut* out,
                            std::string* err) {
    if (!out) {
        if (err) *err = "BuildNorm2LutPiecewise: null output.";
        return false;
    }
    *out = Norm2Lut{};
    if (n < 0 || requested_centers <= 0) {
        if (err) *err = "BuildNorm2LutPiecewise: invalid dims.";
        return false;
    }
    if (n == 0) {
        return true;
    }

    const int k = std::min(std::max(1, requested_centers), std::min(n, 256));
    if (k <= 1 || n < 3) {
        return BuildNorm2Lut(data, n, requested_centers, 1, out, err);
    }

    std::vector<float> sorted(static_cast<std::size_t>(n));
    std::copy(data, data + n, sorted.begin());
    std::sort(sorted.begin(), sorted.end());
    const double q1 = std::clamp(piecewise_p1, 0.0, 1.0);
    const double q2 = std::clamp(piecewise_p2, q1, 1.0);
    const float p99 = SortedQuantileValue(sorted, q1);
    const float p999 = SortedQuantileValue(sorted, q2);
    const auto it1 = std::upper_bound(sorted.begin(), sorted.end(), p99);
    const auto it2 = std::upper_bound(it1, sorted.end(), p999);
    const int b0 = static_cast<int>(it1 - sorted.begin());
    const int b1 = static_cast<int>(it2 - sorted.begin());
    const std::vector<int> seg_begin = {0, b0, b1};
    const std::vector<int> seg_end = {b0, b1, n};
    std::vector<int> counts(3, 0);
    std::vector<float> mins(3, 0.0f);
    std::vector<float> maxs(3, 0.0f);
    for (int s = 0; s < 3; ++s) {
        const int lo = seg_begin[static_cast<std::size_t>(s)];
        const int hi = seg_end[static_cast<std::size_t>(s)];
        counts[static_cast<std::size_t>(s)] = std::max(0, hi - lo);
        if (hi > lo) {
            mins[static_cast<std::size_t>(s)] = sorted[static_cast<std::size_t>(lo)];
            maxs[static_cast<std::size_t>(s)] = sorted[static_cast<std::size_t>(hi - 1)];
        }
    }

    const double count_weight = std::max(0.0, piecewise_count_weight);
    const double range_weight = std::max(0.0, piecewise_range_weight);
    const std::vector<int> bins = AllocatePiecewiseBins(counts, mins, maxs, k, count_weight, range_weight);
    out->codes_u8.resize(static_cast<std::size_t>(n), 0);
    std::vector<int> seg_offset(3, 0);
    for (int s = 1; s < 3; ++s) {
        seg_offset[static_cast<std::size_t>(s)] =
            seg_offset[static_cast<std::size_t>(s - 1)] + bins[static_cast<std::size_t>(s - 1)];
    }
    out->centers.reserve(static_cast<std::size_t>(seg_offset[2] + bins[2]));
    for (int s = 0; s < 3; ++s) {
        const int lo = seg_begin[static_cast<std::size_t>(s)];
        const int hi = seg_end[static_cast<std::size_t>(s)];
        const int local_n = std::max(0, hi - lo);
        const int local_k = std::min(bins[static_cast<std::size_t>(s)], local_n);
        if (local_n <= 0 || local_k <= 0) continue;
        for (int b = 0; b < local_k; ++b) {
            const int bin_lo = lo + static_cast<int>((static_cast<long long>(b) * local_n) / local_k);
            const int bin_hi = lo + static_cast<int>((static_cast<long long>(b + 1) * local_n) / local_k);
            const int clamped_hi = std::max(bin_lo + 1, bin_hi);
            double sum = 0.0;
            for (int i = bin_lo; i < clamped_hi; ++i) {
                sum += static_cast<double>(sorted[static_cast<std::size_t>(i)]);
            }
            out->centers.push_back(
                static_cast<float>(sum / static_cast<double>(std::max(1, clamped_hi - bin_lo))));
        }
    }
    if (out->centers.empty()) {
        return BuildNorm2Lut(data, n, requested_centers, 1, out, err);
    }

    #pragma omp parallel for schedule(static) default(none) shared(data, out, seg_offset, bins) firstprivate(n, p99, p999)
    for (int i = 0; i < n; ++i) {
        const float x = data[i];
        int seg = 0;
        if (x > p999) {
            seg = 2;
        } else if (x > p99) {
            seg = 1;
        }
        const int offset = seg_offset[static_cast<std::size_t>(seg)];
        const int local_k = bins[static_cast<std::size_t>(seg)];
        if (local_k <= 0) {
            (*out).codes_u8[static_cast<std::size_t>(i)] = 0;
            continue;
        }
        float best = std::fabs(x - out->centers[static_cast<std::size_t>(offset)]);
        int best_k = 0;
        for (int j = 1; j < local_k; ++j) {
            const float dist = std::fabs(x - out->centers[static_cast<std::size_t>(offset + j)]);
            const bool closer = (dist < best);
            best = closer ? dist : best;
            best_k = closer ? j : best_k;
        }
        out->codes_u8[static_cast<std::size_t>(i)] =
            static_cast<std::uint8_t>(offset + best_k);
    }
    return true;
}

bool ReadSourceHashForLutCache(const io::LinkageListReader& linkage_list,
                               bool use_coeff_codec,
                               std::uint64_t* out_hash) {
    if (!out_hash) return false;
    return app::ReadNorm2SourceHash(linkage_list.dir(), use_coeff_codec, out_hash);
}

std::uint64_t ComputeNorm2LutCacheHash(const io::LinkageListReader& linkage_list,
                                       bool use_coeff_codec,
                                       int requested_centers,
                                       int kmeans_niter,
                                       const char* mode_tag,
                                       const std::string& disk_norm2_mode,
                                       double log1p_alpha,
                                       double piecewise_p1,
                                       double piecewise_p2,
                                       double piecewise_count_weight,
                                       double piecewise_range_weight) {
    std::uint64_t source_hash = 0;
    if (!ReadSourceHashForLutCache(linkage_list, use_coeff_codec, &source_hash)) {
        return 0;
    }
    Hash64 hh;
    hh.AddStr("linkage_norm2_lut_cache_v1");
    hh.AddStr(mode_tag ? std::string(mode_tag) : std::string());
    hh.AddStr(disk_norm2_mode);
    hh.AddU64(source_hash);
    hh.AddI32(std::max(1, requested_centers));
    hh.AddI32(std::max(1, kmeans_niter));
    if (ResolveNorm2LutTransform(disk_norm2_mode) == Norm2LutTransform::kLog1p) {
        hh.AddBytes(&log1p_alpha, sizeof(log1p_alpha));
    }
    if (disk_norm2_mode == "lut_piecewise") {
        hh.AddBytes(&piecewise_p1, sizeof(piecewise_p1));
        hh.AddBytes(&piecewise_p2, sizeof(piecewise_p2));
        hh.AddBytes(&piecewise_count_weight, sizeof(piecewise_count_weight));
        hh.AddBytes(&piecewise_range_weight, sizeof(piecewise_range_weight));
    }
    return hh.h;
}

}  // namespace

bool IsDiskNorm2ModeLut(const std::string& mode) {
    return IsDiskNorm2ModeLutGlobal(mode) || IsDiskNorm2ModeLutCluster(mode);
}

bool IsDiskNorm2ModeLutGlobal(const std::string& mode) {
    return mode == "lut" || mode == "lut_sqrt" || mode == "lut_log1p" ||
           mode == "lut_tail_weighted" || mode == "lut_piecewise";
}

bool IsDiskNorm2ModeLutCluster(const std::string& mode) {
    return mode == "lut_cluster";
}

bool IsDiskNorm2ModeLutSqrt(const std::string& mode) {
    return mode == "lut_sqrt";
}

bool IsDiskNorm2ModeLutLog1p(const std::string& mode) {
    return mode == "lut_log1p";
}

bool BuildNorm2Lut(const float* data,
                   int n,
                   int requested_centers,
                   int max_iter,
                   Norm2Lut* out,
                   std::string* err) {
    if (!out) {
        if (err) *err = "BuildNorm2Lut: null output.";
        return false;
    }
    *out = Norm2Lut{};
    if (n < 0 || requested_centers <= 0) {
        if (err) *err = "BuildNorm2Lut: invalid dims.";
        return false;
    }
    if (n == 0) {
        return true;
    }

    const int k = std::min(std::max(1, requested_centers), std::min(n, 256));

    std::vector<int> assignments;
    std::vector<float> centers;
    Kmeans1DQuantize(data, n, k, max_iter, &assignments, &centers);
    out->centers = std::move(centers);
    out->codes_u8.resize(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        out->codes_u8[static_cast<std::size_t>(i)] =
            static_cast<std::uint8_t>(assignments[static_cast<std::size_t>(i)]);
    }
    return true;
}

bool BuildNorm2LutWithMode(const float* data,
                           int n,
                           int requested_centers,
                           int max_iter,
                           const std::string& mode,
                           double log1p_alpha,
                           double piecewise_p1,
                           double piecewise_p2,
                           double piecewise_count_weight,
                           double piecewise_range_weight,
                           Norm2Lut* out,
                           std::string* err) {
    if (!out) {
        if (err) *err = "BuildNorm2LutWithMode: null output.";
        return false;
    }
    if (mode == "lut_tail_weighted") {
        *out = Norm2Lut{};
        if (n < 0 || requested_centers <= 0) {
            if (err) *err = "BuildNorm2LutWithMode: invalid input.";
            return false;
        }
        if (n == 0) {
            return true;
        }
        const int k = std::min(std::max(1, requested_centers), std::min(n, 256));
        std::vector<int> assignments;
        std::vector<float> centers;
        Kmeans1DQuantizeTailWeighted(data, n, k, max_iter, &assignments, &centers);
        out->centers = std::move(centers);
        out->codes_u8.resize(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            out->codes_u8[static_cast<std::size_t>(i)] =
                static_cast<std::uint8_t>(assignments[static_cast<std::size_t>(i)]);
        }
        return true;
    }
    if (mode == "lut_piecewise") {
        return BuildNorm2LutPiecewise(data, n, requested_centers,
                                      piecewise_p1, piecewise_p2,
                                      piecewise_count_weight, piecewise_range_weight,
                                      out, err);
    }
    const Norm2LutTransform tfm = ResolveNorm2LutTransform(mode);
    if (tfm == Norm2LutTransform::kIdentity) {
        return BuildNorm2Lut(data, n, requested_centers, max_iter, out, err);
    }
    if (n < 0 || (!data && n > 0)) {
        if (err) *err = "BuildNorm2LutWithMode: invalid input.";
        return false;
    }
    std::vector<float> transformed(static_cast<std::size_t>(std::max(0, n)), 0.0f);
    for (int i = 0; i < n; ++i) {
        transformed[static_cast<std::size_t>(i)] =
            ForwardNorm2LutTransform(data[i], tfm, log1p_alpha);
    }
    if (!BuildNorm2Lut(transformed.data(), n, requested_centers, max_iter, out, err)) {
        return false;
    }
    for (float& c : out->centers) {
        c = InverseNorm2LutTransform(c, tfm, log1p_alpha);
    }
    return true;
}

std::string DiskNorm2LutCodesFilename(bool use_coeff_codec) {
    return use_coeff_codec ? "norm2_int8_lut_codes.u8"
                           : "norm2_float_lut_codes.u8";
}

std::string DiskNorm2LutCentersFilename(bool use_coeff_codec) {
    return use_coeff_codec ? "norm2_int8_lut_centers.f32"
                           : "norm2_float_lut_centers.f32";
}

std::string DiskNorm2LutHashFilename(bool use_coeff_codec) {
    return use_coeff_codec ? "norm2_int8_lut.hash.u64"
                           : "norm2_float_lut.hash.u64";
}

std::string DiskNorm2LutClusterCodesFilename(bool use_coeff_codec) {
    return use_coeff_codec ? "norm2_int8_lut_cluster_codes.u8"
                           : "norm2_float_lut_cluster_codes.u8";
}

std::string DiskNorm2LutClusterCentersFilename(bool use_coeff_codec) {
    return use_coeff_codec ? "norm2_int8_lut_cluster_centers.f32"
                           : "norm2_float_lut_cluster_centers.f32";
}

std::string DiskNorm2LutClusterCenterOffsetsFilename(bool use_coeff_codec) {
    return use_coeff_codec ? "norm2_int8_lut_cluster_center_offsets.u64"
                           : "norm2_float_lut_cluster_center_offsets.u64";
}

std::string DiskNorm2LutClusterHashFilename(bool use_coeff_codec) {
    return use_coeff_codec ? "norm2_int8_lut_cluster.hash.u64"
                           : "norm2_float_lut_cluster.hash.u64";
}

Norm2LutDiskLoadResult LoadNorm2LutCache(const io::LinkageListReader& linkage_list,
                                         bool use_coeff_codec,
                                         int requested_centers,
                                         int kmeans_niter,
                                         const std::string& mode,
                                         double log1p_alpha,
                                         double piecewise_p1,
                                         double piecewise_p2,
                                         double piecewise_count_weight,
                                         double piecewise_range_weight,
                                         Norm2Lut* out,
                                         std::string* err) {
    if (!out) return Norm2LutDiskLoadResult::kInvalid;
    *out = Norm2Lut{};

    const std::filesystem::path dir(linkage_list.dir());
    const std::filesystem::path codes_path = dir / DiskNorm2LutCodesFilename(use_coeff_codec);
    const std::filesystem::path centers_path = dir / DiskNorm2LutCentersFilename(use_coeff_codec);
    const auto check_res = CheckNorm2LutCache(linkage_list, use_coeff_codec,
                                              requested_centers, kmeans_niter,
                                              mode, log1p_alpha,
                                              piecewise_p1, piecewise_p2,
                                              piecewise_count_weight, piecewise_range_weight,
                                              err);
    if (check_res != Norm2LutDiskLoadResult::kLoaded) {
        return check_res;
    }

    const std::uint64_t total_real = linkage_list.total_real();
    const auto centers_bytes = std::filesystem::file_size(centers_path);
    const std::uint64_t total_centers = centers_bytes / sizeof(float);

    out->centers.resize(static_cast<std::size_t>(total_centers));
    if (total_centers > 0) {
        std::ifstream fin(centers_path, std::ios::binary);
        if (!fin) {
            if (err) *err = "failed to open " + centers_path.string();
            return Norm2LutDiskLoadResult::kInvalid;
        }
        fin.read(reinterpret_cast<char*>(out->centers.data()),
                 static_cast<std::streamsize>(out->centers.size() * sizeof(float)));
        if (!fin) {
            if (err) *err = "failed to read " + centers_path.string();
            return Norm2LutDiskLoadResult::kInvalid;
        }
    }

    out->codes_u8.resize(static_cast<std::size_t>(total_real));
    if (total_real > 0) {
        std::ifstream fin(codes_path, std::ios::binary);
        if (!fin) {
            if (err) *err = "failed to open " + codes_path.string();
            return Norm2LutDiskLoadResult::kInvalid;
        }
        fin.read(reinterpret_cast<char*>(out->codes_u8.data()),
                 static_cast<std::streamsize>(out->codes_u8.size()));
        if (!fin) {
            if (err) *err = "failed to read " + codes_path.string();
            return Norm2LutDiskLoadResult::kInvalid;
        }
    }
    return Norm2LutDiskLoadResult::kLoaded;
}

Norm2LutDiskLoadResult LoadNorm2LutCache(const io::LinkageListReader& linkage_list,
                                         bool use_coeff_codec,
                                         int requested_centers,
                                         int kmeans_niter,
                                         Norm2Lut* out,
                                         std::string* err) {
    return LoadNorm2LutCache(linkage_list, use_coeff_codec, requested_centers, kmeans_niter,
                             "lut", 1.0, 0.99, 0.999, 0.5, 0.5, out, err);
}

Norm2LutDiskLoadResult CheckNorm2LutCache(const io::LinkageListReader& linkage_list,
                                          bool use_coeff_codec,
                                          int requested_centers,
                                          int kmeans_niter,
                                          const std::string& mode,
                                          double log1p_alpha,
                                          double piecewise_p1,
                                          double piecewise_p2,
                                          double piecewise_count_weight,
                                          double piecewise_range_weight,
                                          std::string* err) {
    const std::filesystem::path dir(linkage_list.dir());
    const std::filesystem::path codes_path = dir / DiskNorm2LutCodesFilename(use_coeff_codec);
    const std::filesystem::path centers_path = dir / DiskNorm2LutCentersFilename(use_coeff_codec);
    const std::filesystem::path hash_path = dir / DiskNorm2LutHashFilename(use_coeff_codec);

    std::error_code fec;
    if (!std::filesystem::exists(codes_path, fec) || fec) {
        return Norm2LutDiskLoadResult::kMissing;
    }
    if (!std::filesystem::exists(centers_path, fec) || fec) {
        if (err) *err = "missing LUT centers file";
        return Norm2LutDiskLoadResult::kInvalid;
    }

    const std::uint64_t total_real = linkage_list.total_real();
    const std::uint64_t expected_codes_bytes = total_real;
    const auto actual_codes_bytes = std::filesystem::file_size(codes_path, fec);
    if (fec || actual_codes_bytes != expected_codes_bytes) {
        if (err) *err = "unexpected LUT codes payload size";
        return Norm2LutDiskLoadResult::kInvalid;
    }

    const auto centers_bytes = std::filesystem::file_size(centers_path, fec);
    if (fec || (centers_bytes % sizeof(float)) != 0) {
        if (err) *err = "unexpected LUT centers payload size";
        return Norm2LutDiskLoadResult::kInvalid;
    }
    std::uint64_t cache_hash = 0;
    if (!app::ReadU64File(hash_path.string(), &cache_hash)) {
        if (err) *err = "missing " + hash_path.filename().string();
        return Norm2LutDiskLoadResult::kInvalid;
    }
    const std::uint64_t expected_hash =
        ComputeNorm2LutCacheHash(linkage_list, use_coeff_codec, requested_centers, kmeans_niter, "lut",
                                 mode, log1p_alpha,
                                 piecewise_p1, piecewise_p2,
                                 piecewise_count_weight, piecewise_range_weight);
    if (expected_hash == 0) {
        if (err) *err = use_coeff_codec ? "missing linkage_list/hash.u64 or coeff_hash.u64"
                                        : "missing linkage_list/hash.u64";
        return Norm2LutDiskLoadResult::kInvalid;
    }
    if (cache_hash != expected_hash) {
        if (err) {
            *err = "norm2 LUT hash mismatch (check eval.disk_norm2_mode, "
                   "eval.disk_norm2_lut_kmeans_niter, eval.disk_norm2_lut_log_alpha, and piecewise params)";
        }
        return Norm2LutDiskLoadResult::kInvalid;
    }
    return Norm2LutDiskLoadResult::kLoaded;
}

Norm2LutDiskLoadResult CheckNorm2LutCache(const io::LinkageListReader& linkage_list,
                                          bool use_coeff_codec,
                                          int requested_centers,
                                          int kmeans_niter,
                                          std::string* err) {
    return CheckNorm2LutCache(linkage_list, use_coeff_codec, requested_centers, kmeans_niter,
                              "lut", 1.0, 0.99, 0.999, 0.5, 0.5, err);
}

bool StoreNorm2LutCache(const io::LinkageListReader& linkage_list,
                        bool use_coeff_codec,
                        int requested_centers,
                        int kmeans_niter,
                        const std::string& mode,
                        double log1p_alpha,
                        double piecewise_p1,
                        double piecewise_p2,
                        double piecewise_count_weight,
                        double piecewise_range_weight,
                        const Norm2Lut& lut_cache,
                        std::string* err) {
    const std::uint64_t total_real = linkage_list.total_real();
    if (total_real == 0) {
        return true;
    }
    if (lut_cache.codes_u8.size() != static_cast<std::size_t>(total_real)) {
        if (err) *err = "StoreNorm2LutCache: unexpected global LUT codes size.";
        return false;
    }

    const std::filesystem::path dir(linkage_list.dir());
    const std::filesystem::path codes_path = dir / DiskNorm2LutCodesFilename(use_coeff_codec);
    const std::filesystem::path centers_path = dir / DiskNorm2LutCentersFilename(use_coeff_codec);

    {
        std::ofstream fout(codes_path, std::ios::binary);
        if (!fout) {
            if (err) *err = "StoreNorm2LutCache: failed to open " + codes_path.string();
            return false;
        }
        fout.write(reinterpret_cast<const char*>(lut_cache.codes_u8.data()),
                   static_cast<std::streamsize>(lut_cache.codes_u8.size()));
        if (!fout) {
            if (err) *err = "StoreNorm2LutCache: failed to write " + codes_path.string();
            return false;
        }
    }
    {
        std::ofstream fout(centers_path, std::ios::binary);
        if (!fout) {
            if (err) *err = "StoreNorm2LutCache: failed to open " + centers_path.string();
            return false;
        }
        fout.write(reinterpret_cast<const char*>(lut_cache.centers.data()),
                   static_cast<std::streamsize>(lut_cache.centers.size() * sizeof(float)));
        if (!fout) {
            if (err) *err = "StoreNorm2LutCache: failed to write " + centers_path.string();
            return false;
        }
    }
    const std::uint64_t cache_hash =
        ComputeNorm2LutCacheHash(linkage_list, use_coeff_codec, requested_centers, kmeans_niter, "lut",
                                 mode, log1p_alpha,
                                 piecewise_p1, piecewise_p2,
                                 piecewise_count_weight, piecewise_range_weight);
    if (cache_hash != 0) {
        std::string hash_err;
        if (!app::WriteU64FileHex((dir / DiskNorm2LutHashFilename(use_coeff_codec)).string(),
                                  cache_hash,
                                  &hash_err)) {
            if (err) *err = hash_err;
            return false;
        }
    }
    return true;
}

bool StoreNorm2LutCache(const io::LinkageListReader& linkage_list,
                        bool use_coeff_codec,
                        int requested_centers,
                        int kmeans_niter,
                        const Norm2Lut& lut_cache,
                        std::string* err) {
    return StoreNorm2LutCache(linkage_list, use_coeff_codec, requested_centers, kmeans_niter,
                              "lut", 1.0, 0.99, 0.999, 0.5, 0.5, lut_cache, err);
}

bool LoadNorm2LutCenters(const io::LinkageListReader& linkage_list,
                         bool use_coeff_codec,
                         int requested_centers,
                         int kmeans_niter,
                         const std::string& mode,
                         double log1p_alpha,
                         double piecewise_p1,
                         double piecewise_p2,
                         double piecewise_count_weight,
                         double piecewise_range_weight,
                         std::vector<float>* out,
                         std::string* err) {
    if (!out) {
        if (err) *err = "LoadNorm2LutCenters: null output.";
        return false;
    }
    out->clear();
    const auto check_res = CheckNorm2LutCache(linkage_list, use_coeff_codec,
                                              requested_centers, kmeans_niter,
                                              mode, log1p_alpha,
                                              piecewise_p1, piecewise_p2,
                                              piecewise_count_weight, piecewise_range_weight,
                                              err);
    if (check_res != Norm2LutDiskLoadResult::kLoaded) {
        return false;
    }
    const std::filesystem::path centers_path =
        std::filesystem::path(linkage_list.dir()) / DiskNorm2LutCentersFilename(use_coeff_codec);
    const auto centers_bytes = std::filesystem::file_size(centers_path);
    out->resize(static_cast<std::size_t>(centers_bytes / sizeof(float)));
    if (out->empty()) {
        return true;
    }
    std::ifstream fin(centers_path, std::ios::binary);
    if (!fin) {
        if (err) *err = "LoadNorm2LutCenters: failed to open " + centers_path.string();
        out->clear();
        return false;
    }
    fin.read(reinterpret_cast<char*>(out->data()),
             static_cast<std::streamsize>(out->size() * sizeof(float)));
    if (!fin) {
        if (err) *err = "LoadNorm2LutCenters: failed to read " + centers_path.string();
        out->clear();
        return false;
    }
    return true;
}

bool LoadNorm2LutCenters(const io::LinkageListReader& linkage_list,
                         bool use_coeff_codec,
                         int requested_centers,
                         int kmeans_niter,
                         std::vector<float>* out,
                         std::string* err) {
    return LoadNorm2LutCenters(linkage_list, use_coeff_codec, requested_centers, kmeans_niter,
                               "lut", 1.0, 0.99, 0.999, 0.5, 0.5, out, err);
}

Norm2LutDiskLoadResult LoadNorm2LutClusterCache(const io::LinkageListReader& linkage_list,
                                                bool use_coeff_codec,
                                                int requested_centers,
                                                int kmeans_niter,
                                                std::unordered_map<int, Norm2Lut>* out,
                                                std::string* err) {
    if (!out) return Norm2LutDiskLoadResult::kInvalid;
    out->clear();

    const std::filesystem::path dir(linkage_list.dir());
    const std::filesystem::path codes_path = dir / DiskNorm2LutClusterCodesFilename(use_coeff_codec);
    const std::filesystem::path centers_path = dir / DiskNorm2LutClusterCentersFilename(use_coeff_codec);
    const std::filesystem::path center_offsets_path = dir / DiskNorm2LutClusterCenterOffsetsFilename(use_coeff_codec);
    const auto check_res = CheckNorm2LutClusterCache(linkage_list, use_coeff_codec,
                                                     requested_centers, kmeans_niter, err);
    if (check_res != Norm2LutDiskLoadResult::kLoaded) {
        return check_res;
    }

    const auto& real_offs = linkage_list.real_offsets();
    const int nlist = linkage_list.nlist();
    const std::uint64_t total_real = linkage_list.total_real();
    std::vector<std::uint64_t> center_offsets(static_cast<std::size_t>(nlist + 1), 0);
    {
        std::ifstream fin(center_offsets_path, std::ios::binary);
        if (!fin) {
            if (err) *err = "failed to open " + center_offsets_path.string();
            return Norm2LutDiskLoadResult::kInvalid;
        }
        fin.read(reinterpret_cast<char*>(center_offsets.data()),
                 static_cast<std::streamsize>(center_offsets.size() * sizeof(std::uint64_t)));
        if (!fin) {
            if (err) *err = "failed to read " + center_offsets_path.string();
            return Norm2LutDiskLoadResult::kInvalid;
        }
    }

    const std::uint64_t total_centers = center_offsets.back();
    std::vector<float> centers_flat(static_cast<std::size_t>(total_centers));
    if (total_centers > 0) {
        std::ifstream fin(centers_path, std::ios::binary);
        if (!fin) {
            if (err) *err = "failed to open " + centers_path.string();
            return Norm2LutDiskLoadResult::kInvalid;
        }
        fin.read(reinterpret_cast<char*>(centers_flat.data()),
                 static_cast<std::streamsize>(centers_flat.size() * sizeof(float)));
        if (!fin) {
            if (err) *err = "failed to read " + centers_path.string();
            return Norm2LutDiskLoadResult::kInvalid;
        }
    }

    std::vector<std::uint8_t> codes_flat(static_cast<std::size_t>(total_real));
    if (total_real > 0) {
        std::ifstream fin(codes_path, std::ios::binary);
        if (!fin) {
            if (err) *err = "failed to open " + codes_path.string();
            return Norm2LutDiskLoadResult::kInvalid;
        }
        fin.read(reinterpret_cast<char*>(codes_flat.data()),
                 static_cast<std::streamsize>(codes_flat.size()));
        if (!fin) {
            if (err) *err = "failed to read " + codes_path.string();
            return Norm2LutDiskLoadResult::kInvalid;
        }
    }

    out->reserve(static_cast<std::size_t>(nlist));
    for (int cid = 0; cid < nlist; ++cid) {
        const auto lo = real_offs[static_cast<std::size_t>(cid)];
        const auto hi = real_offs[static_cast<std::size_t>(cid + 1)];
        const auto nr = hi - lo;
        if (nr == 0) continue;
        const auto clo = center_offsets[static_cast<std::size_t>(cid)];
        const auto chi = center_offsets[static_cast<std::size_t>(cid + 1)];
        if (chi < clo || chi > total_centers) {
            if (err) *err = "invalid LUT center offsets";
            out->clear();
            return Norm2LutDiskLoadResult::kInvalid;
        }
        Norm2Lut lut;
        lut.codes_u8.resize(static_cast<std::size_t>(nr));
        lut.centers.resize(static_cast<std::size_t>(chi - clo));
        std::memcpy(lut.codes_u8.data(), codes_flat.data() + static_cast<std::size_t>(lo),
                    static_cast<std::size_t>(nr));
        if (chi > clo) {
            std::memcpy(lut.centers.data(),
                        centers_flat.data() + static_cast<std::size_t>(clo),
                        static_cast<std::size_t>(chi - clo) * sizeof(float));
        }
        out->emplace(cid, std::move(lut));
    }

    return Norm2LutDiskLoadResult::kLoaded;
}

Norm2LutDiskLoadResult CheckNorm2LutClusterCache(const io::LinkageListReader& linkage_list,
                                                 bool use_coeff_codec,
                                                 int requested_centers,
                                                 int kmeans_niter,
                                                 std::string* err) {
    const std::filesystem::path dir(linkage_list.dir());
    const std::filesystem::path codes_path = dir / DiskNorm2LutClusterCodesFilename(use_coeff_codec);
    const std::filesystem::path centers_path = dir / DiskNorm2LutClusterCentersFilename(use_coeff_codec);
    const std::filesystem::path center_offsets_path = dir / DiskNorm2LutClusterCenterOffsetsFilename(use_coeff_codec);
    const std::filesystem::path hash_path = dir / DiskNorm2LutClusterHashFilename(use_coeff_codec);

    std::error_code fec;
    if (!std::filesystem::exists(codes_path, fec) || fec) {
        return Norm2LutDiskLoadResult::kMissing;
    }
    if (!std::filesystem::exists(centers_path, fec) || fec ||
        !std::filesystem::exists(center_offsets_path, fec) || fec) {
        if (err) *err = "missing LUT cluster centers/offsets file";
        return Norm2LutDiskLoadResult::kInvalid;
    }

    const auto& real_offs = linkage_list.real_offsets();
    const int nlist = linkage_list.nlist();
    const std::uint64_t total_real = linkage_list.total_real();
    if (real_offs.size() != static_cast<std::size_t>(nlist + 1)) {
        if (err) *err = "invalid real_offsets size";
        return Norm2LutDiskLoadResult::kInvalid;
    }

    const std::uint64_t expected_codes_bytes = total_real;
    const auto actual_codes_bytes = std::filesystem::file_size(codes_path, fec);
    if (fec || actual_codes_bytes != expected_codes_bytes) {
        if (err) *err = "unexpected LUT cluster codes payload size";
        return Norm2LutDiskLoadResult::kInvalid;
    }

    const auto center_offsets_bytes = std::filesystem::file_size(center_offsets_path, fec);
    if (fec || center_offsets_bytes != static_cast<std::uint64_t>(nlist + 1) * sizeof(std::uint64_t)) {
        if (err) *err = "unexpected LUT cluster center_offsets payload size";
        return Norm2LutDiskLoadResult::kInvalid;
    }

    std::uint64_t total_centers = 0;
    std::ifstream fin(center_offsets_path, std::ios::binary);
    if (!fin) {
        if (err) *err = "failed to open " + center_offsets_path.string();
        return Norm2LutDiskLoadResult::kInvalid;
    }
    fin.seekg(static_cast<std::streamoff>(nlist) * static_cast<std::streamoff>(sizeof(std::uint64_t)),
              std::ios::beg);
    if (!fin) {
        if (err) *err = "failed to seek " + center_offsets_path.string();
        return Norm2LutDiskLoadResult::kInvalid;
    }
    fin.read(reinterpret_cast<char*>(&total_centers), static_cast<std::streamsize>(sizeof(std::uint64_t)));
    if (!fin) {
        if (err) *err = "failed to read tail of " + center_offsets_path.string();
        return Norm2LutDiskLoadResult::kInvalid;
    }

    const auto centers_bytes = std::filesystem::file_size(centers_path, fec);
    if (fec || centers_bytes != total_centers * sizeof(float)) {
        if (err) *err = "unexpected LUT cluster centers payload size";
        return Norm2LutDiskLoadResult::kInvalid;
    }
    std::uint64_t cache_hash = 0;
    if (!app::ReadU64File(hash_path.string(), &cache_hash)) {
        if (err) *err = "missing " + hash_path.filename().string();
        return Norm2LutDiskLoadResult::kInvalid;
    }
    const std::uint64_t expected_hash =
        ComputeNorm2LutCacheHash(linkage_list, use_coeff_codec, requested_centers, kmeans_niter,
                                 "lut_cluster", "lut_cluster", 1.0,
                                 0.99, 0.999, 0.5, 0.5);
    if (expected_hash == 0) {
        if (err) *err = use_coeff_codec ? "missing linkage_list/hash.u64 or coeff_hash.u64"
                                        : "missing linkage_list/hash.u64";
        return Norm2LutDiskLoadResult::kInvalid;
    }
    if (cache_hash != expected_hash) {
        if (err) *err = "norm2 LUT hash mismatch (check eval.disk_norm2_lut_kmeans_niter)";
        return Norm2LutDiskLoadResult::kInvalid;
    }
    return Norm2LutDiskLoadResult::kLoaded;
}

bool StoreNorm2LutClusterCache(const io::LinkageListReader& linkage_list,
                               bool use_coeff_codec,
                               int requested_centers,
                               int kmeans_niter,
                               const std::unordered_map<int, Norm2Lut>& lut_cache,
                               std::string* err) {
    const auto& real_offs = linkage_list.real_offsets();
    const int nlist = linkage_list.nlist();
    const std::uint64_t total_real = linkage_list.total_real();
    if (real_offs.size() != static_cast<std::size_t>(nlist + 1) || total_real == 0) {
        return true;
    }

    std::vector<std::uint64_t> center_offsets(static_cast<std::size_t>(nlist + 1), 0);
    std::vector<float> centers_flat;
    std::vector<std::uint8_t> codes_u8(static_cast<std::size_t>(total_real), 0);

    for (int cid = 0; cid < nlist; ++cid) {
        center_offsets[static_cast<std::size_t>(cid)] = static_cast<std::uint64_t>(centers_flat.size());
        const auto lo = real_offs[static_cast<std::size_t>(cid)];
        const auto hi = real_offs[static_cast<std::size_t>(cid + 1)];
        const auto nr = hi - lo;
        if (nr == 0) continue;
        const auto it = lut_cache.find(cid);
        if (it == lut_cache.end()) continue;
        const Norm2Lut& lut = it->second;
        const std::size_t nr_sz = static_cast<std::size_t>(nr);
        if (lut.centers.empty() || lut.codes_u8.size() != nr_sz) continue;
        centers_flat.insert(centers_flat.end(), lut.centers.begin(), lut.centers.end());
        std::memcpy(codes_u8.data() + static_cast<std::size_t>(lo),
                    lut.codes_u8.data(),
                    nr_sz);
    }
    center_offsets[static_cast<std::size_t>(nlist)] = static_cast<std::uint64_t>(centers_flat.size());

    const std::filesystem::path dir(linkage_list.dir());
    const std::filesystem::path codes_path = dir / DiskNorm2LutClusterCodesFilename(use_coeff_codec);
    const std::filesystem::path centers_path = dir / DiskNorm2LutClusterCentersFilename(use_coeff_codec);
    const std::filesystem::path center_offsets_path = dir / DiskNorm2LutClusterCenterOffsetsFilename(use_coeff_codec);

    {
        std::ofstream fout(codes_path, std::ios::binary);
        if (!fout) {
            if (err) *err = "StoreNorm2LutClusterCache: failed to open " + codes_path.string();
            return false;
        }
        fout.write(reinterpret_cast<const char*>(codes_u8.data()),
                   static_cast<std::streamsize>(codes_u8.size()));
        if (!fout) {
            if (err) *err = "StoreNorm2LutClusterCache: failed to write " + codes_path.string();
            return false;
        }
    }
    {
        std::ofstream fout(centers_path, std::ios::binary);
        if (!fout) {
            if (err) *err = "StoreNorm2LutClusterCache: failed to open " + centers_path.string();
            return false;
        }
        fout.write(reinterpret_cast<const char*>(centers_flat.data()),
                   static_cast<std::streamsize>(centers_flat.size() * sizeof(float)));
        if (!fout) {
            if (err) *err = "StoreNorm2LutClusterCache: failed to write " + centers_path.string();
            return false;
        }
    }
    {
        std::ofstream fout(center_offsets_path, std::ios::binary);
        if (!fout) {
            if (err) *err = "StoreNorm2LutClusterCache: failed to open " + center_offsets_path.string();
            return false;
        }
        fout.write(reinterpret_cast<const char*>(center_offsets.data()),
                   static_cast<std::streamsize>(center_offsets.size() * sizeof(std::uint64_t)));
        if (!fout) {
            if (err) *err = "StoreNorm2LutClusterCache: failed to write " + center_offsets_path.string();
            return false;
        }
    }
    const std::uint64_t cache_hash =
        ComputeNorm2LutCacheHash(linkage_list, use_coeff_codec, requested_centers, kmeans_niter,
                                 "lut_cluster", "lut_cluster", 1.0,
                                 0.99, 0.999, 0.5, 0.5);
    if (cache_hash != 0) {
        std::string hash_err;
        if (!app::WriteU64FileHex((dir / DiskNorm2LutClusterHashFilename(use_coeff_codec)).string(),
                                  cache_hash,
                                  &hash_err)) {
            if (err) *err = hash_err;
            return false;
        }
    }
    return true;
}

bool LoadFloatNorm2ClusterSlice(const io::LinkageListReader& linkage_list,
                                bool use_coeff_codec,
                                int cid,
                                std::vector<float>* out,
                                std::string* err) {
    if (!out) {
        if (err) *err = "LoadFloatNorm2ClusterSlice: null output.";
        return false;
    }
    out->clear();
    const auto& real_offs = linkage_list.real_offsets();
    const int nlist = linkage_list.nlist();
    if (cid < 0 || cid >= nlist || real_offs.size() != static_cast<std::size_t>(nlist + 1)) {
        if (err) *err = "LoadFloatNorm2ClusterSlice: invalid cid.";
        return false;
    }
    const std::uint64_t lo = real_offs[static_cast<std::size_t>(cid)];
    const std::uint64_t hi = real_offs[static_cast<std::size_t>(cid + 1)];
    const std::uint64_t nr = hi - lo;
    if (nr == 0) {
        return true;
    }

    const std::filesystem::path path = std::filesystem::path(linkage_list.dir()) /
                                       (use_coeff_codec ? "norm2_int8.f32" : "norm2_float.f32");
    std::ifstream fin(path, std::ios::binary);
    if (!fin) {
        if (err) *err = "LoadFloatNorm2ClusterSlice: failed to open " + path.string();
        return false;
    }
    out->resize(static_cast<std::size_t>(nr));
    fin.seekg(static_cast<std::streamoff>(lo * sizeof(float)), std::ios::beg);
    if (!fin) {
        if (err) *err = "LoadFloatNorm2ClusterSlice: failed to seek " + path.string();
        out->clear();
        return false;
    }
    fin.read(reinterpret_cast<char*>(out->data()),
             static_cast<std::streamsize>(nr * sizeof(float)));
    if (!fin) {
        if (err) *err = "LoadFloatNorm2ClusterSlice: failed to read " + path.string();
        out->clear();
        return false;
    }
    return true;
}

bool LoadNorm2LutClusterCodesSlice(const io::LinkageListReader& linkage_list,
                                   bool use_coeff_codec,
                                   int cid,
                                   std::vector<std::uint8_t>* out,
                                   std::string* err) {
    if (!out) {
        if (err) *err = "LoadNorm2LutClusterCodesSlice: null output.";
        return false;
    }
    out->clear();
    const auto& real_offs = linkage_list.real_offsets();
    const int nlist = linkage_list.nlist();
    if (cid < 0 || cid >= nlist || real_offs.size() != static_cast<std::size_t>(nlist + 1)) {
        if (err) *err = "LoadNorm2LutClusterCodesSlice: invalid cid.";
        return false;
    }

    const std::uint64_t lo = real_offs[static_cast<std::size_t>(cid)];
    const std::uint64_t hi = real_offs[static_cast<std::size_t>(cid + 1)];
    const std::uint64_t nr = hi - lo;
    if (nr == 0) {
        return true;
    }

    const std::filesystem::path codes_path =
        std::filesystem::path(linkage_list.dir()) / DiskNorm2LutCodesFilename(use_coeff_codec);
    std::ifstream fin(codes_path, std::ios::binary);
    if (!fin) {
        if (err) *err = "LoadNorm2LutClusterCodesSlice: failed to open " + codes_path.string();
        return false;
    }
    out->resize(static_cast<std::size_t>(nr));
    fin.seekg(static_cast<std::streamoff>(lo), std::ios::beg);
    if (!fin) {
        if (err) *err = "LoadNorm2LutClusterCodesSlice: failed to seek " + codes_path.string();
        out->clear();
        return false;
    }
    fin.read(reinterpret_cast<char*>(out->data()), static_cast<std::streamsize>(nr));
    if (!fin) {
        if (err) *err = "LoadNorm2LutClusterCodesSlice: failed to read " + codes_path.string();
        out->clear();
        return false;
    }
    return true;
}

bool LoadNorm2LutClusterSlice(const io::LinkageListReader& linkage_list,
                              bool use_coeff_codec,
                              int cid,
                              Norm2Lut* out,
                              std::string* err) {
    if (!out) {
        if (err) *err = "LoadNorm2LutClusterSlice: null output.";
        return false;
    }
    *out = Norm2Lut{};
    const auto& real_offs = linkage_list.real_offsets();
    const int nlist = linkage_list.nlist();
    if (cid < 0 || cid >= nlist || real_offs.size() != static_cast<std::size_t>(nlist + 1)) {
        if (err) *err = "LoadNorm2LutClusterSlice: invalid cid.";
        return false;
    }

    const std::uint64_t lo = real_offs[static_cast<std::size_t>(cid)];
    const std::uint64_t hi = real_offs[static_cast<std::size_t>(cid + 1)];
    const std::uint64_t nr = hi - lo;
    if (nr == 0) {
        return true;
    }

    const std::filesystem::path dir(linkage_list.dir());
    const std::filesystem::path codes_path = dir / DiskNorm2LutClusterCodesFilename(use_coeff_codec);
    const std::filesystem::path centers_path = dir / DiskNorm2LutClusterCentersFilename(use_coeff_codec);
    const std::filesystem::path center_offsets_path = dir / DiskNorm2LutClusterCenterOffsetsFilename(use_coeff_codec);

    std::uint64_t clo = 0;
    std::uint64_t chi = 0;
    {
        std::ifstream fin(center_offsets_path, std::ios::binary);
        if (!fin) {
            if (err) *err = "LoadNorm2LutClusterSlice: failed to open " + center_offsets_path.string();
            return false;
        }
        fin.seekg(static_cast<std::streamoff>(cid) * static_cast<std::streamoff>(sizeof(std::uint64_t)),
                  std::ios::beg);
        if (!fin) {
            if (err) *err = "LoadNorm2LutClusterSlice: failed to seek " + center_offsets_path.string();
            return false;
        }
        fin.read(reinterpret_cast<char*>(&clo), static_cast<std::streamsize>(sizeof(std::uint64_t)));
        fin.read(reinterpret_cast<char*>(&chi), static_cast<std::streamsize>(sizeof(std::uint64_t)));
        if (!fin || chi < clo) {
            if (err) *err = "LoadNorm2LutClusterSlice: invalid center offsets in " + center_offsets_path.string();
            return false;
        }
    }

    out->codes_u8.resize(static_cast<std::size_t>(nr));
    {
        std::ifstream fin(codes_path, std::ios::binary);
        if (!fin) {
            if (err) *err = "LoadNorm2LutClusterSlice: failed to open " + codes_path.string();
            *out = Norm2Lut{};
            return false;
        }
        fin.seekg(static_cast<std::streamoff>(lo), std::ios::beg);
        if (!fin) {
            if (err) *err = "LoadNorm2LutClusterSlice: failed to seek " + codes_path.string();
            *out = Norm2Lut{};
            return false;
        }
        fin.read(reinterpret_cast<char*>(out->codes_u8.data()), static_cast<std::streamsize>(nr));
        if (!fin) {
            if (err) *err = "LoadNorm2LutClusterSlice: failed to read " + codes_path.string();
            *out = Norm2Lut{};
            return false;
        }
    }

    const std::uint64_t center_count = chi - clo;
    out->centers.resize(static_cast<std::size_t>(center_count));
    if (center_count > 0) {
        std::ifstream fin(centers_path, std::ios::binary);
        if (!fin) {
            if (err) *err = "LoadNorm2LutClusterSlice: failed to open " + centers_path.string();
            *out = Norm2Lut{};
            return false;
        }
        fin.seekg(static_cast<std::streamoff>(clo * sizeof(float)), std::ios::beg);
        if (!fin) {
            if (err) *err = "LoadNorm2LutClusterSlice: failed to seek " + centers_path.string();
            *out = Norm2Lut{};
            return false;
        }
        fin.read(reinterpret_cast<char*>(out->centers.data()),
                 static_cast<std::streamsize>(center_count * sizeof(float)));
        if (!fin) {
            if (err) *err = "LoadNorm2LutClusterSlice: failed to read " + centers_path.string();
            *out = Norm2Lut{};
            return false;
        }
    }
    return true;
}

}  // namespace stlq::eval
