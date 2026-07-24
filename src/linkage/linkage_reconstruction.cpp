#include "stlq/linkage/linkage_reconstruction.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include <omp.h>

#include "stlq/core/blas.h"
#include "stlq/core/threading.h"
#include "stlq/common/logger.h"

namespace stlq {

namespace {

inline int ClampDepth(int d) {
    return std::max(0, d);
}

std::vector<int> BuildRootToOneMappingFromCenters(const ColMajorMatrix<float>& C_root0,
                                                  const ColMajorMatrix<float>& C_one0) {
    const int d = C_root0.rows;
    const int h0_root = C_root0.cols;
    const int h0_one = C_one0.cols;
    if (d <= 0 || h0_root <= 0 || h0_one <= 0) {
        return {};
    }
    if (C_one0.rows != d) {
        throw std::runtime_error("BuildRootToOneMappingFromCenters: dim mismatch.");
    }

    std::vector<int> root_to_one(static_cast<std::size_t>(h0_root), 0);
    constexpr int kBlock = 4096;
    ColMajorMatrix<float> scores(h0_one, std::min(kBlock, h0_root));

    ScopedBlasThreads blas_scope(OmpMaxThreads());
    for (int r0 = 0; r0 < h0_root; r0 += kBlock) {
        const int rlen = std::min(kBlock, h0_root - r0);
        scores.rows = h0_one;
        scores.cols = rlen;
        scores.data.resize(static_cast<std::size_t>(h0_one) * static_cast<std::size_t>(rlen));

        // scores = C_one0' * C_root0_block  (h0_one x rlen)
        GemmRaw(true, false,
                h0_one, rlen, d,
                1.0f,
                C_one0.data.data(), d,
                C_root0.data.data() + static_cast<std::size_t>(r0) * static_cast<std::size_t>(d), d,
                0.0f,
                scores.data.data(), h0_one);

        for (int j = 0; j < rlen; ++j) {
            int best = 0;
            float best_s = scores(0, j);
            for (int k = 1; k < h0_one; ++k) {
                const float s = scores(k, j);
                if (s > best_s) {
                    best_s = s;
                    best = k;
                }
            }
            root_to_one[static_cast<std::size_t>(r0 + j)] = best;
        }
    }
    return root_to_one;
}

void BuildDepthOrder(const std::vector<int>& depth,
                     std::vector<int>* offsets,
                     std::vector<int>* order) {
    const int n = static_cast<int>(depth.size());
    int max_depth = 0;
    for (int i = 0; i < n; ++i) {
        max_depth = std::max(max_depth, ClampDepth(depth[static_cast<std::size_t>(i)]));
    }

    offsets->assign(static_cast<std::size_t>(max_depth + 2), 0);
    for (int i = 0; i < n; ++i) {
        int di = ClampDepth(depth[static_cast<std::size_t>(i)]);
        (*offsets)[static_cast<std::size_t>(di + 1)] += 1;
    }
    for (std::size_t d = 1; d < offsets->size(); ++d) {
        (*offsets)[d] += (*offsets)[d - 1];
    }

    order->assign(static_cast<std::size_t>(n), 0);
    std::vector<int> cursor = *offsets;
    for (int i = 0; i < n; ++i) {
        int di = ClampDepth(depth[static_cast<std::size_t>(i)]);
        int pos = cursor[static_cast<std::size_t>(di)]++;
        (*order)[static_cast<std::size_t>(pos)] = i;
    }
}

}  // namespace

ColMajorMatrix<float> ComputeLinkagedReconstructionMultiBook(const CodebookPack& C_root,
                                                            const CodebookPack& C_one,
                                                            const ColMajorMatrix<FullCode>& B,
                                                            const ColMajorMatrix<float>& a,
                                                            const std::vector<int>& parent,
                                                            const std::vector<int>& depth) {
    const int d = C_root.d;
    const int n = B.cols;
    const int m = static_cast<int>(C_root.books.size());
    if (d <= 0 || n <= 0 || m <= 0) {
        return {};
    }
    if (B.rows != m || a.rows != m || a.cols != n) {
        throw std::runtime_error("ComputeLinkagedReconstructionMultiBook: B/a shape mismatch.");
    }
    if (static_cast<int>(C_one.books.size()) != m) {
        throw std::runtime_error("ComputeLinkagedReconstructionMultiBook: C_one book count mismatch.");
    }
    for (int l = 0; l < m; ++l) {
        if (C_root.books[l].rows != d || C_one.books[l].rows != d) {
            throw std::runtime_error("ComputeLinkagedReconstructionMultiBook: codebook dim mismatch.");
        }
    }
    if (static_cast<int>(parent.size()) != n || static_cast<int>(depth.size()) != n) {
        throw std::runtime_error("ComputeLinkagedReconstructionMultiBook: parent/depth length mismatch.");
    }

    const bool need_root_to_one = (C_root.books[0].cols != C_one.books[0].cols);
    std::vector<int> root_to_one;
    if (need_root_to_one) {
        root_to_one = BuildRootToOneMappingFromCenters(C_root.books[0], C_one.books[0]);
    }

    std::vector<int> offsets;
    std::vector<int> order;
    BuildDepthOrder(depth, &offsets, &order);
    const int max_depth = static_cast<int>(offsets.size()) - 2;

    ColMajorMatrix<float> out(d, n);
    std::fill(out.data.begin(), out.data.end(), 0.0f);

    for (int dep = 0; dep <= max_depth; ++dep) {
        const int begin = offsets[static_cast<std::size_t>(dep)];
        const int end = offsets[static_cast<std::size_t>(dep + 1)];

        #pragma omp parallel for default(none) schedule(static) shared(order, depth, parent, C_root, C_one, B, a, root_to_one, out) firstprivate(begin, end, d, m, need_root_to_one)
        for (int pos = begin; pos < end; ++pos) {
            const int i = order[static_cast<std::size_t>(pos)];
            const auto& books = (ClampDepth(depth[static_cast<std::size_t>(i)]) == 0)
                                    ? C_root.books
                                    : C_one.books;

            float* dst = out.Col(i);
            std::fill(dst, dst + d, 0.0f);
            for (int l = 0; l < m; ++l) {
                int code = static_cast<int>(B(l, i));
                if (need_root_to_one && ClampDepth(depth[static_cast<std::size_t>(i)]) > 0 && l == 0) {
                    code = root_to_one[static_cast<std::size_t>(code)];
                }
                // if (code < 0 || code >= books[l].cols) continue;
                float coeff = a(l, i);
                const float* center = books[l].Col(code);
                #pragma omp simd
                for (int r = 0; r < d; ++r) {
                    dst[r] += coeff * center[r];
                }
            }

            int p = parent[static_cast<std::size_t>(i)];
            if (p != 0) {
                const int p_idx = p - 1;
                const float* rp = out.Col(p_idx);
                #pragma omp simd
                for (int r = 0; r < d; ++r) {
                    dst[r] += rp[r];
                }
            }
        }
    }
    return out;
}

ColMajorMatrix<float> ComputeLinkagedReconstructionMultiBook(const CodebookPack& C_root,
                                                            const CodebookPack& C_one,
                                                            const ColMajorMatrix<FullCode>& B,
                                                            const ColMajorMatrix<float>& a,
                                                            const LinkageStructure& linkage) {
    const int d = C_root.d;
    const int n = B.cols;
    ColMajorMatrix<float> out(d, n);
    std::fill(out.data.begin(), out.data.end(), 0.0f);
    ApplyLinkagedReconstructionMultiBookInPlace(C_root, C_one, B, a, linkage, &out);
    return out;
}

void ApplyLinkagedReconstructionMultiBookInPlace(const CodebookPack& C_root,
                                                const CodebookPack& C_one,
                                                const ColMajorMatrix<FullCode>& B,
                                                const ColMajorMatrix<float>& a,
                                                const LinkageStructure& linkage,
                                                ColMajorMatrix<float>* out) {
    const int d = C_root.d;
    const int n = B.cols;
    const int m = static_cast<int>(C_root.books.size());
    if (!out) {
        throw std::runtime_error("ApplyLinkagedReconstructionMultiBookInPlace: null out.");
    }
    if (d <= 0 || n <= 0 || m <= 0) {
        return;
    }
    if (out->rows != d || out->cols != n) {
        throw std::runtime_error("ApplyLinkagedReconstructionMultiBookInPlace: out shape mismatch.");
    }
    if (B.rows != m || a.rows != m || a.cols != n) {
        throw std::runtime_error("ApplyLinkagedReconstructionMultiBookInPlace: B/a shape mismatch.");
    }
    if (static_cast<int>(C_one.books.size()) != m) {
        throw std::runtime_error("ApplyLinkagedReconstructionMultiBookInPlace: C_one book count mismatch.");
    }
    for (int l = 0; l < m; ++l) {
        if (C_root.books[l].rows != d || C_one.books[l].rows != d) {
            throw std::runtime_error("ApplyLinkagedReconstructionMultiBookInPlace: codebook dim mismatch.");
        }
    }

    const bool need_root_to_one = (C_root.books[0].cols != C_one.books[0].cols);
    std::vector<int> root_to_one;
    if (need_root_to_one) {
        root_to_one = BuildRootToOneMappingFromCenters(C_root.books[0], C_one.books[0]);
    }

    #pragma omp parallel for default(none) schedule(dynamic) shared(linkage, C_root, C_one, B, a, out, root_to_one) firstprivate(d, m, n, need_root_to_one)
    for (int c = 0; c < static_cast<int>(linkage.clusters.size()); ++c) {
        const auto& cluster = linkage.clusters[static_cast<std::size_t>(c)];
        const int csize = static_cast<int>(cluster.indices.size());
        if (csize <= 0) {
            continue;
        }

        if (cluster.depth_offsets.size() < 2) {
            LogWarn("ApplyLinkagedRecon[c=" + std::to_string(c) + "]: depth_offsets.size()=" +
                    std::to_string(cluster.depth_offsets.size()) + " <2 for non-empty cluster; skipping.");
            continue;
        }
        const int n_real = (cluster.n_real > 0) ? std::min(cluster.n_real, csize) : csize;
        if (n_real <= 0) {
            continue;
        }
        const int max_depth = static_cast<int>(cluster.depth_offsets.size()) - 2;
        for (int dep = 0; dep <= max_depth; ++dep) {
            const int begin = cluster.depth_offsets[static_cast<std::size_t>(dep)];
            const int end = cluster.depth_offsets[static_cast<std::size_t>(dep + 1)];
            for (int local = begin; local < end; ++local) {
                if (local < 0 || local >= n_real) {
                    LogWarn("ApplyLinkagedRecon[c=" + std::to_string(c) + " dep=" + std::to_string(dep) +
                            "]: local=" + std::to_string(local) + " out of range [0," +
                            std::to_string(n_real) + "); linkage data corrupt.");
                    continue;
                }
                const int g = cluster.indices[static_cast<std::size_t>(local)];
                const auto& books = (dep == 0) ? C_root.books : C_one.books;

                float* dst = out->Col(static_cast<int>(g));
                std::fill(dst, dst + d, 0.0f);
                for (int l = 0; l < m; ++l) {
                    int code = static_cast<int>(B(l, static_cast<int>(g)));
                    if (need_root_to_one && dep > 0 && l == 0) {
                        code = root_to_one[static_cast<std::size_t>(code)];
                    }
                    float coeff = a(l, static_cast<int>(g));
                    const float* center = books[l].Col(code);
                    #pragma omp simd
                    for (int r = 0; r < d; ++r) {
                        dst[r] += coeff * center[r];
                    }
                }

                const int p_local = cluster.parent_local[static_cast<std::size_t>(local)];
                if (p_local != 0) {
                    const int p_idx_local = static_cast<int>(p_local - 1);
                    const int p_global = cluster.indices[static_cast<std::size_t>(p_idx_local)];
                    const float* rp = out->Col(static_cast<int>(p_global));
                    #pragma omp simd
                    for (int r = 0; r < d; ++r) {
                        dst[r] += rp[r];
                    }
                }
            }
        }
    }
}

void ApplyLinkagedReconstructionMultiBookInPlacePtrs(const CodebookPack& C_root,
                                                    const CodebookPack& C_one,
                                                    const FullCode* const* B_cols,
                                                    const float* const* a_cols,
                                                    int n,
                                                    const LinkageStructure& linkage,
                                                    float* const* out_cols) {
    const int d = C_root.d;
    const int m = static_cast<int>(C_root.books.size());
    if (!B_cols || !a_cols || !out_cols) {
        throw std::runtime_error("ApplyLinkagedReconstructionMultiBookInPlacePtrs: null pointers.");
    }
    if (d <= 0 || n <= 0 || m <= 0) {
        return;
    }
    if (static_cast<int>(C_one.books.size()) != m) {
        throw std::runtime_error("ApplyLinkagedReconstructionMultiBookInPlacePtrs: C_one book count mismatch.");
    }
    for (int l = 0; l < m; ++l) {
        if (C_root.books[l].rows != d || C_one.books[l].rows != d) {
            throw std::runtime_error("ApplyLinkagedReconstructionMultiBookInPlacePtrs: codebook dim mismatch.");
        }
    }

    const bool need_root_to_one = (C_root.books[0].cols != C_one.books[0].cols);
    std::vector<int> root_to_one;
    if (need_root_to_one) {
        root_to_one = BuildRootToOneMappingFromCenters(C_root.books[0], C_one.books[0]);
    }

    #pragma omp parallel for default(none) schedule(dynamic) shared(linkage, C_root, C_one, B_cols, a_cols, out_cols, root_to_one) firstprivate(d, m, n, need_root_to_one)
    for (int c = 0; c < static_cast<int>(linkage.clusters.size()); ++c) {
        const auto& cluster = linkage.clusters[static_cast<std::size_t>(c)];
        const int csize = static_cast<int>(cluster.indices.size());
        if (csize <= 0) {
            continue;
        }

        if (cluster.depth_offsets.size() < 2) {
            LogWarn("ApplyLinkagedReconPtrs[c=" + std::to_string(c) + "]: depth_offsets.size()=" +
                    std::to_string(cluster.depth_offsets.size()) + " <2 for non-empty cluster; skipping.");
            continue;
        }
        const int n_real = (cluster.n_real > 0) ? std::min(cluster.n_real, csize) : csize;
        if (n_real <= 0) {
            continue;
        }
        const int max_depth = static_cast<int>(cluster.depth_offsets.size()) - 2;
        for (int dep = 0; dep <= max_depth; ++dep) {
            const int begin = cluster.depth_offsets[static_cast<std::size_t>(dep)];
            const int end = cluster.depth_offsets[static_cast<std::size_t>(dep + 1)];
            for (int local = begin; local < end; ++local) {
                if (local < 0 || local >= n_real) {
                    LogWarn("ApplyLinkagedReconPtrs[c=" + std::to_string(c) + " dep=" + std::to_string(dep) +
                            "]: local=" + std::to_string(local) + " out of range [0," +
                            std::to_string(n_real) + "); linkage data corrupt.");
                    continue;
                }
                const int g = cluster.indices[static_cast<std::size_t>(local)];
                if (g < 0 || g >= n) {
                    throw std::runtime_error("ApplyLinkagedReconstructionMultiBookInPlacePtrs: index out of range.");
                }

                const auto& books = (dep == 0) ? C_root.books : C_one.books;

                float* dst = out_cols[g];
                std::fill(dst, dst + d, 0.0f);
                const FullCode* bcol = B_cols[g];
                const float* acol = a_cols[g];
                for (int l = 0; l < m; ++l) {
                    int code = static_cast<int>(bcol[static_cast<std::size_t>(l)]);
                    if (need_root_to_one && dep > 0 && l == 0) {
                        code = root_to_one[static_cast<std::size_t>(code)];
                    }
                    const float coeff = acol[static_cast<std::size_t>(l)];
                    const float* center = books[l].Col(code);
                    #pragma omp simd
                    for (int r = 0; r < d; ++r) {
                        dst[r] += coeff * center[r];
                    }
                }

                const int p_local = cluster.parent_local[static_cast<std::size_t>(local)];
                if (p_local != 0) {
                    const int p_idx_local = static_cast<int>(p_local - 1);
                    const int p_global = cluster.indices[static_cast<std::size_t>(p_idx_local)];
                    const float* rp = out_cols[p_global];
                    #pragma omp simd
                    for (int r = 0; r < d; ++r) {
                        dst[r] += rp[r];
                    }
                }
            }
        }
    }
}

ColMajorMatrix<float> ComputeLinkagedReconstructionMultiBook(const CodebookPack& C_root,
                                                            const CodebookPack& C_one,
                                                            const FullCode* const* B_cols,
                                                            const float* const* a_cols,
                                                            int n,
                                                            const LinkageStructure& linkage) {
    const int d = C_root.d;
    const int m = static_cast<int>(C_root.books.size());
    if (d <= 0 || n <= 0 || m <= 0) {
        return {};
    }
    if (!B_cols || !a_cols) {
        throw std::runtime_error("ComputeLinkagedReconstructionMultiBook(cols): null column pointers.");
    }
    if (static_cast<int>(C_one.books.size()) != m) {
        throw std::runtime_error("ComputeLinkagedReconstructionMultiBook(cols): C_one book count mismatch.");
    }
    for (int l = 0; l < m; ++l) {
        if (C_root.books[l].rows != d || C_one.books[l].rows != d) {
            throw std::runtime_error("ComputeLinkagedReconstructionMultiBook(cols): codebook dim mismatch.");
        }
    }

    ColMajorMatrix<float> out(d, n);
    std::fill(out.data.begin(), out.data.end(), 0.0f);
    std::vector<float*> out_cols(static_cast<std::size_t>(n), nullptr);
    for (int i = 0; i < n; ++i) {
        out_cols[static_cast<std::size_t>(i)] = out.Col(i);
    }
    ApplyLinkagedReconstructionMultiBookInPlacePtrs(C_root, C_one,
                                                   B_cols, a_cols, n,
                                                   linkage,
                                                   out_cols.data());
    return out;
}

}  // namespace stlq
