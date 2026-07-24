#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "stlq/eval/recall_linkage_scan_cuda.h"
#include "stlq/common/types.h"

namespace {

inline bool BetterPair(float da, std::uint32_t ia, float db, std::uint32_t ib) {
    if (da < db) return true;
    if (da > db) return false;
    return ia < ib;
}

std::vector<std::pair<float, std::uint32_t>> CpuTopKLookupLinkagedScaledTables(
    const stlq::STLQueryTables& qt,
    int qi,
    int m,
    int m_codes,
    const int* offsets_root_small,
    const int* offsets_one,
    const stlq::eval::ClusterView& cv,
    int k,
    std::vector<std::pair<float, std::uint32_t>>* all_items_out = nullptr) {
    const int n_real = cv.n_real;
    const int n_virt = cv.n_virt;
    const int nc = cv.nc;
    const int real_base = n_virt;
    const int n_root_real = cv.n_root_real;
    const std::uint32_t* real_ids = cv.real_ids;
    const std::uint32_t* parent_1based = cv.parent_1based;
    const std::uint32_t* depth_offsets = cv.depth_offsets;
    const std::uint8_t* codes_small = cv.codes_small_bytes;
    const std::uint8_t* code0_one = cv.code0_one_bytes;
    const std::uint8_t* virt_codes_small = cv.virt_codes_small_bytes;
    const std::int8_t* q_layer_major = cv.q_layer_major;
    const float* scales_root = cv.scales_root;
    const float* scales_linkage = cv.scales_linkage;
    const float* r_norm2 = cv.r_norm2;

    std::vector<float> dot_all(static_cast<std::size_t>(nc), 0.0f);

    const float* xCq_root0 = qt.xCq_root0.Col(qi);
    const float* xCq_root_small = qt.xCq_root_small.Col(qi);
    const float* xCq_one = qt.xCq_one.Col(qi);

    const float root0_scaled = scales_root[0] * xCq_root0[cv.cid];
    const auto stride_codes = static_cast<std::ptrdiff_t>(m_codes);

    // Virtual + real roots.
    for (int v = 0; v < n_virt; ++v) {
        float dot = root0_scaled * static_cast<float>(q_layer_major[v]);
        const std::uint8_t* row = virt_codes_small + static_cast<std::ptrdiff_t>(v) * stride_codes;
        for (int l = 1; l < m; ++l) {
            const std::int8_t* ql = q_layer_major + static_cast<std::ptrdiff_t>(l) * nc;
            const float sl = scales_root[l];
            const float* table = xCq_root_small + offsets_root_small[l];
            const int code = static_cast<int>(row[l - 1]);
            dot += static_cast<float>(ql[v]) * (sl * table[code]);
        }
        dot_all[v] = dot;
    }
    for (int pos = 0; pos < n_root_real; ++pos) {
        const int local = real_base + pos;
        float dot = root0_scaled * static_cast<float>(q_layer_major[local]);
        const std::uint8_t* row = codes_small + static_cast<std::ptrdiff_t>(pos) * stride_codes;
        for (int l = 1; l < m; ++l) {
            const std::int8_t* ql = q_layer_major + static_cast<std::ptrdiff_t>(l) * nc;
            const float sl = scales_root[l];
            const float* table = xCq_root_small + offsets_root_small[l];
            const int code = static_cast<int>(row[l - 1]);
            dot += static_cast<float>(ql[local]) * (sl * table[code]);
        }
        dot_all[local] = dot;
    }

    // Linkage nodes (depth-parallel).
    const int depth_len = cv.depth_offsets_len;
    for (int dep = 1; dep + 1 < depth_len; ++dep) {
        const int begin = static_cast<int>(depth_offsets[dep]);
        const int end = static_cast<int>(depth_offsets[dep + 1]);
        for (int pos = begin; pos < end; ++pos) {
            const int local = real_base + pos;
            const int p = static_cast<int>(parent_1based[pos]) - 1;
            float dot = dot_all[p];

            const int code0 = static_cast<int>(code0_one[pos]);
            dot += static_cast<float>(q_layer_major[local]) *
                   (scales_linkage[0] * xCq_one[offsets_one[0] + code0]);

            const std::uint8_t* row = codes_small + static_cast<std::ptrdiff_t>(pos) * stride_codes;
            for (int l = 1; l < m; ++l) {
                const std::int8_t* ql = q_layer_major + static_cast<std::ptrdiff_t>(l) * nc;
                const float sl = scales_linkage[l];
                const float* one_l = xCq_one + offsets_one[l];
                const int code = static_cast<int>(row[l - 1]);
                dot += static_cast<float>(ql[local]) * (sl * one_l[code]);
            }
            dot_all[local] = dot;
        }
    }

    std::vector<std::pair<float, std::uint32_t>> items;
    items.reserve(static_cast<std::size_t>(n_real));
    for (int pos = 0; pos < n_real; ++pos) {
        const float dist = r_norm2[pos] - 2.0f * dot_all[real_base + pos];
        items.emplace_back(dist, real_ids[pos]);
    }
    if (all_items_out) {
        *all_items_out = items;
    }
    std::sort(items.begin(), items.end(),
              [](const auto& a, const auto& b) { return BetterPair(a.first, a.second, b.first, b.second); });
    if (static_cast<int>(items.size()) > k) items.resize(static_cast<std::size_t>(k));
    return items;
}

}  // namespace

int main() {
    using namespace stlq;
    using stlq::eval::ClusterView;

    // Small deterministic setup to validate GPU scan correctness in cache mode.
    constexpr int nlist = 3;
    constexpr int qlen = 1;
    constexpr int nprobe_cap = 1;
    constexpr int m = 3;
    constexpr int m_codes = m - 1;
    constexpr int Hr = 8;   // per-layer root-small table size
    constexpr int Ho = 8;   // per-layer one table size
    constexpr int root_small_total_cols = Hr * (m - 1);
    constexpr int one_total_cols = Ho * m;
    constexpr int n_virt = 2;
    constexpr int n_real = 50;
    constexpr int nc = n_virt + n_real;
    constexpr int n_root_real = 10;
    constexpr int k = 10;

    const int offsets_root_small[m] = {0, 0, Hr};
    const int offsets_one[m] = {0, Ho, 2 * Ho};

    STLQueryTables qt;
    qt.xCq_root0 = ColMajorMatrix<float>(nlist, qlen);
    qt.xCq_root_small = ColMajorMatrix<float>(root_small_total_cols, qlen);
    qt.xCq_one = ColMajorMatrix<float>(one_total_cols, qlen);

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> uf(-1.0f, 1.0f);
    for (int i = 0; i < nlist; ++i) qt.xCq_root0.data[static_cast<std::size_t>(i)] = uf(rng);
    for (int i = 0; i < root_small_total_cols; ++i) qt.xCq_root_small.data[static_cast<std::size_t>(i)] = uf(rng);
    for (int i = 0; i < one_total_cols; ++i) qt.xCq_one.data[static_cast<std::size_t>(i)] = uf(rng);

    // Build a single active cluster.
    std::vector<std::uint32_t> real_ids(static_cast<std::size_t>(n_real));
    std::vector<std::uint32_t> parent_1based(static_cast<std::size_t>(n_real), 0);
    std::vector<std::uint32_t> depth_offsets = {0u, static_cast<std::uint32_t>(n_root_real), static_cast<std::uint32_t>(n_real)};
    std::vector<std::uint8_t> codes_small(static_cast<std::size_t>(n_real) * static_cast<std::size_t>(m_codes));
    std::vector<std::uint8_t> code0_one(static_cast<std::size_t>(n_real));
    std::vector<std::uint8_t> virt_codes_small(static_cast<std::size_t>(n_virt) * static_cast<std::size_t>(m_codes));
    std::vector<std::int8_t> q_layer_major(static_cast<std::size_t>(m) * static_cast<std::size_t>(nc));
    std::vector<float> scales_root(static_cast<std::size_t>(m));
    std::vector<float> scales_linkage(static_cast<std::size_t>(m));
    std::vector<float> r_norm2(static_cast<std::size_t>(n_real), 0.0f);

    for (int i = 0; i < n_real; ++i) real_ids[static_cast<std::size_t>(i)] = static_cast<std::uint32_t>(1000 + i);
    // Depth1 linkage nodes have parents among roots (virtual + real root0).
    std::uniform_int_distribution<int> parent_root(0, n_virt + n_root_real - 1);
    for (int pos = n_root_real; pos < n_real; ++pos) {
        const int p_local = parent_root(rng);
        parent_1based[static_cast<std::size_t>(pos)] = static_cast<std::uint32_t>(p_local + 1);
    }

    std::uniform_int_distribution<int> code_root_small(0, Hr - 1);
    std::uniform_int_distribution<int> code_one(0, Ho - 1);
    for (int i = 0; i < n_real * m_codes; ++i) codes_small[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(code_root_small(rng));
    for (int i = 0; i < n_real; ++i) code0_one[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(code_one(rng));
    for (int i = 0; i < n_virt * m_codes; ++i) virt_codes_small[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(code_root_small(rng));

    std::uniform_int_distribution<int> qv(-3, 3);
    for (int i = 0; i < m * nc; ++i) q_layer_major[static_cast<std::size_t>(i)] = static_cast<std::int8_t>(qv(rng));

    for (int l = 0; l < m; ++l) {
        scales_root[static_cast<std::size_t>(l)] = 0.3f + 0.1f * static_cast<float>(l);
        scales_linkage[static_cast<std::size_t>(l)] = 0.2f + 0.05f * static_cast<float>(l);
    }

    ClusterView cv;
    cv.cid = 1;
    cv.m = m;
    cv.m_codes = m_codes;
    cv.n_real = n_real;
    cv.n_virt = n_virt;
    cv.nc = nc;
    cv.n_root_real = n_root_real;
    cv.depth_offsets_len = static_cast<int>(depth_offsets.size());
    cv.real_ids = real_ids.data();
    cv.parent_1based = parent_1based.data();
    cv.depth_offsets = depth_offsets.data();
    cv.codes_small_bytes = codes_small.data();
    cv.code0_one_bytes = code0_one.data();
    cv.virt_codes_small_bytes = virt_codes_small.data();
    cv.q_layer_major = q_layer_major.data();
    cv.scales_root = scales_root.data();
    cv.scales_linkage = scales_linkage.data();
    cv.r_norm2 = r_norm2.data();

    std::vector<ClusterView> active_views;
    active_views.push_back(cv);

    std::vector<int> task_cluster_idx(static_cast<std::size_t>(qlen) * static_cast<std::size_t>(nprobe_cap), 0);

    // CPU reference.
    std::vector<std::pair<float, std::uint32_t>> cpu_all;
    const auto cpu = CpuTopKLookupLinkagedScaledTables(qt, /*qi=*/0, m, m_codes,
                                                      offsets_root_small,
                                                      offsets_one,
                                                      cv, k, &cpu_all);
    std::vector<float> cpu_dist_by_pos(static_cast<std::size_t>(n_real), 0.0f);
    std::vector<std::uint32_t> cpu_id_by_pos(static_cast<std::size_t>(n_real), 0u);
    for (std::size_t i = 0; i < cpu_all.size(); ++i) {
        cpu_dist_by_pos[i] = cpu_all[i].first;
        cpu_id_by_pos[i] = cpu_all[i].second;
    }

    auto run_gpu = [&](int cache_mb, std::vector<float>* dists, std::vector<std::uint32_t>* ids,
                       stlq::eval::cuda::DiskLinkageGpuScanStats* stats) -> bool {
        std::string err;

        return stlq::eval::cuda::ScanDiskLinkageLookupTopK(
            qt, m, m_codes, nlist,
            root_small_total_cols, one_total_cols,
            offsets_root_small, offsets_one,
            active_views, task_cluster_idx,
            qlen, nprobe_cap,
            k, /*max_nc=*/64, cache_mb,
            /*tile256=*/false,
            dists, ids, stats, &err);
        };

    std::vector<float> gpu_dists_nocache;
    std::vector<std::uint32_t> gpu_ids_nocache;
    stlq::eval::cuda::DiskLinkageGpuScanStats stats_nocache;
    if (!run_gpu(/*cache_mb=*/0, &gpu_dists_nocache, &gpu_ids_nocache, &stats_nocache)) {
        std::cerr << "test_cuda_linkage_scan_lookup_topk: GPU scan(no-cache) failed\n";
        return 1;
    }

    std::vector<float> gpu_dists_cache;
    std::vector<std::uint32_t> gpu_ids_cache;
    stlq::eval::cuda::DiskLinkageGpuScanStats stats_cache;
    if (!run_gpu(/*cache_mb=*/1, &gpu_dists_cache, &gpu_ids_cache, &stats_cache)) {
        std::cerr << "test_cuda_linkage_scan_lookup_topk: GPU scan(cache) failed\n";
        return 1;
    }

    auto check = [&](const char* tag, const std::vector<float>& gd, const std::vector<std::uint32_t>& gi) -> int {
        if (gi.size() != static_cast<std::size_t>(k) || gd.size() != static_cast<std::size_t>(k)) return k;
        int mismatches = 0;
        for (int j = 0; j < k; ++j) {
            const std::uint32_t id = gi[static_cast<std::size_t>(j)];
            const float dist = gd[static_cast<std::size_t>(j)];
            const std::uint32_t ci = cpu[static_cast<std::size_t>(j)].second;
            const float cd = cpu[static_cast<std::size_t>(j)].first;
            if (id != ci || std::fabs(dist - cd) > 1e-3f) {
                ++mismatches;
                if (mismatches <= 3) {
                    // Find CPU dist for this id (linear scan over n_real is fine for this test).
                    float cpu_d_for_id = std::numeric_limits<float>::quiet_NaN();
                    for (std::size_t ii = 0; ii < cpu_id_by_pos.size(); ++ii) {
                        if (cpu_id_by_pos[ii] == id) {
                            cpu_d_for_id = cpu_dist_by_pos[ii];
                            break;
                        }
                    }
                    std::cerr << "  [" << tag << "] j=" << j << " gpu(id=" << id << " d=" << dist
                              << ") cpu(id=" << ci << " d=" << cd << ") cpu_d_for_gpu_id=" << cpu_d_for_id << "\n";
                }
            }
        }
        if (mismatches != 0) {
            std::cerr << "test_cuda_linkage_scan_lookup_topk: " << tag << " mismatches=" << mismatches << "\n";
        }
        return mismatches;
    };

    const int mm0 = check("no-cache", gpu_dists_nocache, gpu_ids_nocache);
    const int mm1 = check("cache", gpu_dists_cache, gpu_ids_cache);
    if (mm0 != 0 || mm1 != 0) {
        std::cerr << "CPU first 5:\n";
        for (int j = 0; j < 5 && j < k; ++j) {
            std::cerr << "  j=" << j << " id=" << cpu[static_cast<std::size_t>(j)].second
                      << " d=" << cpu[static_cast<std::size_t>(j)].first << "\n";
        }
        std::cerr << "GPU(no-cache) first 5:\n";
        for (int j = 0; j < 5 && j < k; ++j) {
            std::cerr << "  j=" << j << " id=" << gpu_ids_nocache[static_cast<std::size_t>(j)]
                      << " d=" << gpu_dists_nocache[static_cast<std::size_t>(j)] << "\n";
        }
        return 1;
    }

    std::cout << "test_cuda_linkage_scan_lookup_topk: ok (tasks_gpu=" << stats_cache.tasks_gpu
              << ", cache_enabled=" << stats_cache.cache_enabled << ")\n";
    return 0;
}
