#include "stlq/linkage/linkage_summary.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>

#include <omp.h>

#include "stlq/quantizer/linear_algebra.h"

namespace stlq {

std::vector<int> ComputeDepthsFromParent(const std::vector<int>& parent) {
    const int n = static_cast<int>(parent.size());
    std::vector<int> depth(static_cast<std::size_t>(n), -1);
    std::vector<int> path;
    path.reserve(64);
    for (int i = 0; i < n; ++i) {
        if (depth[static_cast<std::size_t>(i)] >= 0) {
            continue;
        }
        path.clear();
        int v = i;
        while (v >= 0 && v < n && depth[static_cast<std::size_t>(v)] < 0) {
            path.push_back(v);
            const int p1 = parent[static_cast<std::size_t>(v)];
            if (p1 == 0) {
                v = -1;
                break;
            }
            v = p1 - 1;
        }
        const int base = (v < 0) ? 0 : (depth[static_cast<std::size_t>(v)] + 1);
        int cur = base;
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            depth[static_cast<std::size_t>(*it)] = cur++;
        }
    }
    for (int i = 0; i < n; ++i) {
        if (depth[static_cast<std::size_t>(i)] < 0) {
            depth[static_cast<std::size_t>(i)] = 0;
        }
    }
    return depth;
}

void SquaredErrorsFromRecon(const ColMajorMatrix<float>& X,
                            const ColMajorMatrix<float>& R_full,
                            std::vector<float>* out) {
    const int d = X.rows;
    const int n = X.cols;
    if (!out) {
        return;
    }
    out->resize(static_cast<std::size_t>(std::max(0, n)));
    if (d <= 0 || n <= 0 || R_full.rows != d || R_full.cols != n) {
        return;
    }

    #pragma omp parallel for default(none) schedule(static) shared(X, R_full, out) firstprivate(d, n)
    for (int i = 0; i < n; ++i) {
        const float* xi = X.Col(i);
        const float* ri = R_full.Col(i);
        float err = 0.0f;
        #pragma omp simd reduction(+:err)
        for (int r = 0; r < d; ++r) {
            const float diff = xi[r] - ri[r];
            err += diff * diff;
        }
        (*out)[static_cast<std::size_t>(i)] = err;
    }
}

MseStats MseStatsFromRecon(const ColMajorMatrix<float>& X,
                           const ColMajorMatrix<float>& R_full) {
    MseStats out;
    const int d = X.rows;
    const int n = X.cols;
    if (d <= 0 || n <= 0 || R_full.rows != d || R_full.cols != n) {
        return out;
    }

    double sum = 0.0;
    double mn = std::numeric_limits<double>::infinity();
    double mx = 0.0;

    #pragma omp parallel default(none) if (!omp_in_parallel()) shared(X, R_full, sum, mn, mx) firstprivate(d, n)
    {
        double sum_local = 0.0;
        double mn_local = std::numeric_limits<double>::infinity();
        double mx_local = 0.0;

        #pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            const float* xi = X.Col(i);
            const float* ri = R_full.Col(i);
            double mse = 0.0;
            #pragma omp simd reduction(+:mse)
            for (int r = 0; r < d; ++r) {
                double diff = static_cast<double>(xi[r]) - static_cast<double>(ri[r]);
                mse += diff * diff;
            }
            sum_local += mse;
            mn_local = std::min(mn_local, mse);
            mx_local = std::max(mx_local, mse);
        }

        #pragma omp atomic
        sum += sum_local;
        #pragma omp critical
        {
            mn = std::min(mn, mn_local);
            mx = std::max(mx, mx_local);
        }
    }

    out.mean = static_cast<float>(sum / static_cast<double>(n));
    out.min = static_cast<float>(mn);
    out.max = static_cast<float>(mx);
    return out;
}

float MeanMseFromRecon(const ColMajorMatrix<float>& X,
                       const ColMajorMatrix<float>& R_full) {
    return MseStatsFromRecon(X, R_full).mean;
}

float MeanMseFromCodebooks(const ColMajorMatrix<float>& X,
                           const std::vector<ColMajorMatrix<float>>& C,
                           const ColMajorMatrix<FullCode>& B,
                           const ColMajorMatrix<float>& a) {
    const int d = X.rows;
    const int n = X.cols;
    const int m = static_cast<int>(C.size());
    if (n <= 0 || d <= 0 || m <= 0 || B.cols != n || a.cols != n) {
        return 0.0f;
    }

    double sum = 0.0;
    #pragma omp parallel default(none) shared(X, C, B, a, sum) firstprivate(d, n, m)
    {
        std::vector<float> recon(static_cast<std::size_t>(d), 0.0f);
        #pragma omp for reduction(+:sum) schedule(static)
        for (int i = 0; i < n; ++i) {
            std::fill(recon.begin(), recon.end(), 0.0f);
            for (int l = 0; l < m; ++l) {
                const int code = static_cast<int>(B(l, i));
                const float coeff = a(l, i);
                const float* center = C[l].Col(code);
                #pragma omp simd
                for (int r = 0; r < d; ++r) {
                    recon[static_cast<std::size_t>(r)] += coeff * center[r];
                }
            }
            const float* xi = X.Col(i);
            double err = 0.0;
            #pragma omp simd reduction(+:err)
            for (int r = 0; r < d; ++r) {
                const double diff =
                    static_cast<double>(xi[r]) - static_cast<double>(recon[static_cast<std::size_t>(r)]);
                err += diff * diff;
            }
            sum += err;
        }
    }
    return static_cast<float>(sum / static_cast<double>(n));
}

float MeanMseFromPrecomp(const ColMajorMatrix<float>& X,
                         const Precomp& precomp,
                         const ColMajorMatrix<float>& xC,
                         const ColMajorMatrix<FullCode>& B,
                         const ColMajorMatrix<float>& a) {
    // Uses ||x||^2 - 2*sum_j a_j<x,c_j> + sum_{j,k} a_j a_k <c_j,c_k>
    // to avoid reconstructing vectors. Valid only if xC/G correspond to precomp/C_all.
    const int d = X.rows;
    const int n = X.cols;
    const int m = precomp.m;
    if (n <= 0 || d <= 0 || xC.cols != n || B.cols != n || a.cols != n) {
        return 0.0f;
    }
    double sum = 0.0;
    #pragma omp parallel for default(none) reduction(+:sum) schedule(static) shared(X, precomp, xC, B, a) firstprivate(d, n, m)
    for (int i = 0; i < n; ++i) {
        const float* xi = X.Col(i);
        double norm_x2 = 0.0;
        #pragma omp simd reduction(+:norm_x2)
        for (int r = 0; r < d; ++r) {
            const double v = xi[r];
            norm_x2 += v * v;
        }

        int flats[64];
        double coeffs[64];
        for (int j = 0; j < m; ++j) {
            flats[j] = precomp.offsets[j] + static_cast<int>(B(j, i));
            coeffs[j] = static_cast<double>(a(j, i));
        }

        double term1 = 0.0;
        for (int j = 0; j < m; ++j) {
            term1 += coeffs[j] * static_cast<double>(xC(flats[j], i));
        }

        double term2 = 0.0;
        for (int j = 0; j < m; ++j) {
            for (int k = 0; k < m; ++k) {
                term2 += coeffs[j] * coeffs[k] *
                         static_cast<double>(GAt(precomp.G, flats[j], flats[k]));
            }
        }

        double err = norm_x2 - 2.0 * term1 + term2;
        if (err < 0.0) {
            err = 0.0;
        }
        sum += err;
    }
    return static_cast<float>(sum / static_cast<double>(n));
}

LinkageDepthStats ComputeLinkageDepthStats(const LinkageStructure& linkage, int n_total) {
    LinkageDepthStats out;
    long long total = (n_total > 0) ? static_cast<long long>(n_total) : 0LL;
    if (total <= 0) {
        for (const auto& cluster : linkage.clusters) {
            const int csize = static_cast<int>(cluster.indices.size());
            const int n_real = (cluster.n_real > 0) ? std::min(cluster.n_real, csize) : csize;
            total += static_cast<long long>(n_real);
        }
    }
    if (total <= 0) {
        return out;
    }

    long long linkaged = 0;
    long long depth_sum = 0;
    int depth_max = 0;

    #pragma omp parallel default(none) if (!omp_in_parallel()) shared(linkage, linkaged, depth_sum, depth_max) firstprivate(total)
    {
        long long linkaged_local = 0;
        long long depth_sum_local = 0;
        int depth_max_local = 0;

        #pragma omp for schedule(static)
        for (int c = 0; c < static_cast<int>(linkage.clusters.size()); ++c) {
            const auto& cluster = linkage.clusters[static_cast<std::size_t>(c)];
        const int csize = static_cast<int>(cluster.indices.size());
        const int n_real = (cluster.n_real > 0) ? std::min(cluster.n_real, csize) : csize;
        if (n_real <= 0) {
            continue;
        }
        if (cluster.depth_offsets.size() >= 2) {
            const int max_depth = static_cast<int>(cluster.depth_offsets.size()) - 2;
            depth_max_local = std::max(depth_max_local, max_depth);
            const int depth0 = cluster.depth_offsets[1] - cluster.depth_offsets[0];
            linkaged_local += static_cast<long long>(n_real - depth0);
            for (int d = 0; d <= max_depth; ++d) {
                const int cnt = cluster.depth_offsets[static_cast<std::size_t>(d + 1)] -
                                cluster.depth_offsets[static_cast<std::size_t>(d)];
                depth_sum_local += static_cast<long long>(d) * static_cast<long long>(cnt);
            }
        } else {
            // Fallback: count linkaged by parent!=0, depth_sum unknown (treat as 0).
            for (int local = 0; local < n_real; ++local) {
                linkaged_local += (cluster.parent_local[static_cast<std::size_t>(local)] != 0) ? 1 : 0;
            }
        }
    }

        #pragma omp atomic
        linkaged += linkaged_local;
        #pragma omp atomic
        depth_sum += depth_sum_local;
        #pragma omp critical
        { depth_max = std::max(depth_max, depth_max_local); }
    }

    out.linkage_ratio = static_cast<float>(static_cast<double>(linkaged) / static_cast<double>(total));
    out.max_depth = depth_max;
    out.mean_depth = static_cast<float>(static_cast<double>(depth_sum) / static_cast<double>(total));
    return out;
}

}  // namespace stlq
