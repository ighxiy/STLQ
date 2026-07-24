#include "stlq/ivf/ivf_scan.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "stlq/linkage/linkage_builder.h"
#include "stlq/linkage/linkage_finalize.h"
#include "stlq/core/blas.h"
#include "stlq/common/types.h"
#include "stlq/ivf/cluster_select.h"
#include "stlq/common/logger.h"
#include "stlq/quantizer/codebook_meta.h"
#include "stlq/succinct/parent_louds.h"
#include "stlq/common/timer.h"

#include <omp.h>

#if defined(STLQ_USE_MKL)
#include <mkl.h>
#else
#include <cblas.h>
#endif

extern "C" {

void quantize_norms_complete_mkl_uint8(
    uint8_t* B,
    const float* a,
    float** codebooks,
    int* codebook_sizes,
    int M,
    int n_base,
    int d,
    int h_norms,
    int max_iter,
    int* assignments,
    float* norm_centers);

void quantize_norms_complete_mkl_uint16(
    uint16_t* B,
    const float* a,
    float** codebooks,
    int* codebook_sizes,
    int M,
    int n_base,
    int d,
    int h_norms,
    int max_iter,
    int* assignments,
    float* norm_centers);

}  // extern "C"

namespace stlq {

namespace {

std::vector<float> ComputeIvfInvNormFromFirstBook(const CodebookMeta& meta, int nlist) {
    std::vector<float> inv_norm(static_cast<std::size_t>(std::max(0, nlist)), 1.0f);
    if (nlist <= 0 || meta.ptrs.empty() || meta.sizes.empty() || meta.d <= 0) {
        return inv_norm;
    }
    const float* book0 = meta.ptrs[0];
    const int cols0 = meta.sizes[0];
    const int limit = std::min(nlist, cols0);
    for (int cid = 0; cid < limit; ++cid) {
        const float* col = book0 + static_cast<std::size_t>(cid) * meta.d;
        double ss = 0.0;
        for (int r = 0; r < meta.d; ++r) {
            const double v = static_cast<double>(col[r]);
            ss += v * v;
        }
        inv_norm[static_cast<std::size_t>(cid)] =
            1.0f / std::sqrt(std::max(static_cast<float>(ss), 1e-20f));
    }
    return inv_norm;
}

static_assert(sizeof(int32_t) == sizeof(int), "Expected 32-bit int.");

STLQ_ALWAYS_INLINE int ParentLocal1b(const LinkageStructure::Cluster& cluster,
                                       const int* parent_override_1b,
                                       int local) {
    if (parent_override_1b) {
        return parent_override_1b[static_cast<std::size_t>(local)];
    }
    return cluster.parent_local[static_cast<std::size_t>(local)];
}

template <typename InCodeT, typename OutCodeT>
bool ConvertCodes(const ColMajorMatrix<InCodeT>& B, std::vector<OutCodeT>* out, std::string* error) {
    out->resize(B.data.size());
    const OutCodeT max_val = std::numeric_limits<OutCodeT>::max();
    for (std::size_t i = 0; i < B.data.size(); ++i) {
        const InCodeT val = B.data[i];
        if (val > max_val) {
            if (error) {
                *error = "Code value exceeds storage type.";
            }
            return false;
        }
        (*out)[i] = static_cast<OutCodeT>(val);
    }
    return true;
}

std::vector<int> BuildDotOffsets(const std::vector<int>& sizes) {
    const int M = static_cast<int>(sizes.size());
    std::vector<int> offs(static_cast<std::size_t>(M) * static_cast<std::size_t>(M), 0);
    int offset = 0;
    for (int i = 0; i < M; ++i) {
        for (int j = i; j < M; ++j) {
            offs[static_cast<std::size_t>(i) * M + j] = offset;
            offs[static_cast<std::size_t>(j) * M + i] = offset;
            offset += sizes[static_cast<std::size_t>(i)] * sizes[static_cast<std::size_t>(j)];
        }
    }
    return offs;
}

int DotTotalSize(const std::vector<int>& sizes) {
    const int M = static_cast<int>(sizes.size());
    int total = 0;
    for (int i = 0; i < M; ++i) {
        for (int j = i; j < M; ++j) {
            total += sizes[static_cast<std::size_t>(i)] * sizes[static_cast<std::size_t>(j)];
        }
    }
    return total;
}

std::vector<int> BuildCrossDotOffsets(const std::vector<int>& sizes_a,
                                      const std::vector<int>& sizes_b) {
    const int ma = static_cast<int>(sizes_a.size());
    const int mb = static_cast<int>(sizes_b.size());
    std::vector<int> offs(static_cast<std::size_t>(ma) * static_cast<std::size_t>(mb), 0);
    int offset = 0;
    for (int i = 0; i < ma; ++i) {
        for (int j = 0; j < mb; ++j) {
            offs[static_cast<std::size_t>(i) * mb + j] = offset;
            offset += sizes_a[static_cast<std::size_t>(i)] * sizes_b[static_cast<std::size_t>(j)];
        }
    }
    return offs;
}

int DotCrossTotalSize(const std::vector<int>& sizes_a,
                      const std::vector<int>& sizes_b) {
    const int ma = static_cast<int>(sizes_a.size());
    const int mb = static_cast<int>(sizes_b.size());
    int total = 0;
    for (int i = 0; i < ma; ++i) {
        for (int j = 0; j < mb; ++j) {
            total += sizes_a[static_cast<std::size_t>(i)] * sizes_b[static_cast<std::size_t>(j)];
        }
    }
    return total;
}

void PrecomputeCodebookDotsCross(const CodebookMeta& meta_a,
                                 const CodebookMeta& meta_b,
                                 const std::vector<int>& cross_offsets,
                                 std::vector<float>* D_out) {
    const int ma = meta_a.m;
    const int mb = meta_b.m;
    const int total = DotCrossTotalSize(meta_a.sizes, meta_b.sizes);
    D_out->assign(static_cast<std::size_t>(total), 0.0f);
    float* D = D_out->data();

    for (int i = 0; i < ma; ++i) {
        for (int j = 0; j < mb; ++j) {
            const int ki = meta_a.sizes[static_cast<std::size_t>(i)];
            const int kj = meta_b.sizes[static_cast<std::size_t>(j)];
            const int off = cross_offsets[static_cast<std::size_t>(i) * mb + j];
            cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                        ki, kj, meta_a.d,
                        1.0f, meta_a.ptrs[static_cast<std::size_t>(i)], meta_a.d,
                              meta_b.ptrs[static_cast<std::size_t>(j)], meta_b.d,
                        0.0f, D + off, ki);
        }
    }
}

void PrecomputeCodebookDots(const CodebookMeta& meta,
                            const std::vector<int>& dot_offsets,
                            std::vector<float>* D_out) {
    const int M = meta.m;
    const int total = DotTotalSize(meta.sizes);
    D_out->assign(static_cast<std::size_t>(total), 0.0f);
    float* D = D_out->data();

    for (int i = 0; i < M; ++i) {
        for (int j = i; j < M; ++j) {
            const int ki = meta.sizes[static_cast<std::size_t>(i)];
            const int kj = meta.sizes[static_cast<std::size_t>(j)];
            const int off = dot_offsets[static_cast<std::size_t>(i) * M + j];
            cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                        ki, kj, meta.d,
                        1.0f, meta.ptrs[static_cast<std::size_t>(i)], meta.d,
                              meta.ptrs[static_cast<std::size_t>(j)], meta.d,
                        0.0f, D + off, ki);
        }
    }
}

inline float DotCodewordPair(const std::vector<int>& sizes,
                             const std::vector<int>& dot_offsets,
                             const float* D,
                             int book_a,
                             int code_a,
                             int book_b,
                             int code_b) {
    int ba = book_a;
    int bb = book_b;
    int ra = code_a;
    int cb = code_b;
    if (ba > bb) {
        std::swap(ba, bb);
        std::swap(ra, cb);
    }
    const int M = static_cast<int>(sizes.size());
    const int lda = sizes[static_cast<std::size_t>(ba)];
    const int off = dot_offsets[static_cast<std::size_t>(ba) * M + bb];
    return D[static_cast<std::size_t>(off) + static_cast<std::size_t>(ra) +
             static_cast<std::size_t>(cb) * static_cast<std::size_t>(lda)];
}

inline float DotCodewordPairCross(const std::vector<int>& sizes_a,
                                  const std::vector<int>& sizes_b,
                                  const std::vector<int>& dot_offsets_ab,
                                  const float* D,
                                  int book_a,
                                  int code_a,
                                  int book_b,
                                  int code_b) {
    const int mb = static_cast<int>(sizes_b.size());
    const int lda = sizes_a[static_cast<std::size_t>(book_a)];
    const int off = dot_offsets_ab[static_cast<std::size_t>(book_a) * mb + book_b];
    return D[static_cast<std::size_t>(off) + static_cast<std::size_t>(code_a) +
             static_cast<std::size_t>(code_b) * static_cast<std::size_t>(lda)];
}

template <typename CodeT>
float DotReconstructionWithOneCodeword_TwoBook(const LinkageStructure::Cluster& cluster,
                                               int start_local,
                                               const int* parent_override_1b,
                                               const CodeT* __restrict codes,
                                               const float* __restrict a,
                                               int m,
                                               const std::vector<int>& sizes_root,
                                               const std::vector<int>& dot_offsets_root_one,
                                               const float* __restrict D_root_one,
                                               const std::vector<int>& sizes_one,
                                               const std::vector<int>& dot_offsets_one,
                                               const float* __restrict D_one,
                                               int target_book_one,
                                               int target_code_one) {
    float acc = 0.0f;
    int cur = start_local;
    while (true) {
        const int g = cluster.indices[static_cast<std::size_t>(cur)];
        const CodeT* __restrict code = codes + static_cast<std::size_t>(g) * m;
        const float* __restrict coeff = a + static_cast<std::size_t>(g) * m;
        const int p_local_1b = ParentLocal1b(cluster, parent_override_1b, cur);
        const bool is_root = (p_local_1b == 0);

        if (!is_root) {
            for (int cb = 0; cb < m; ++cb) {
                acc += coeff[cb] *
                       DotCodewordPair(sizes_one, dot_offsets_one, D_one,
                                       cb, static_cast<int>(code[cb]),
                                       target_book_one, target_code_one);
            }
            cur = p_local_1b - 1;
        } else {
            for (int cb = 0; cb < m; ++cb) {
                acc += coeff[cb] *
                       DotCodewordPairCross(sizes_root, sizes_one, dot_offsets_root_one, D_root_one,
                                            cb, static_cast<int>(code[cb]),
                                            target_book_one, target_code_one);
            }
            break;
        }
    }
    return acc;
}

template <typename CodeT>
float DotReconstructionWithOneCodeword_TwoBookVirtual(const LinkageStructure::Cluster& cluster,
                                                      int start_local,
                                                      const int* parent_override_1b,
                                                      const BaseEncoding& base_real,
                                                      const VirtualEncoding& virt,
                                                      const float* __restrict a_real,
                                                      int m_split,
                                                      const std::vector<int>& sizes_root,
                                                      const std::vector<int>& dot_offsets_root_one,
                                                      const float* __restrict D_root_one,
                                                      const std::vector<int>& sizes_one,
                                                      const std::vector<int>& dot_offsets_one,
                                                      const float* __restrict D_one,
                                                      int target_book_one,
                                                      int target_code_one) {
    float acc = 0.0f;
    int cur = start_local;
    const int cid = cluster.cluster_id;
    const int n_virt = std::max(0, cluster.n_virtual);
    const int n_real_roots = (cluster.depth_offsets.size() >= 2) ? cluster.depth_offsets[1] : 0;
    while (true) {
        const bool is_virtual = (cur < n_virt);
        const bool is_root = is_virtual || ((cur - n_virt) < n_real_roots);

        if (is_virtual) {
            const int v_local = cur;
            const FullCode* __restrict code = virt.B_by_cluster[static_cast<std::size_t>(cid)].Col(v_local);
            const float* __restrict coeff = virt.a_by_cluster[static_cast<std::size_t>(cid)].Col(v_local);
            for (int cb = 0; cb < m_split; ++cb) {
                acc += coeff[cb] *
                       DotCodewordPairCross(sizes_root, sizes_one, dot_offsets_root_one, D_root_one,
                                            cb, static_cast<int>(code[cb]),
                                            target_book_one, target_code_one);
            }
            break;  // virtual nodes are roots
        }

        const int real_idx = cur - n_virt;
        const int g = cluster.indices[static_cast<std::size_t>(real_idx)];
        const FullCode* __restrict code = base_real.B.Col(g);
        const float* __restrict coeff = a_real + static_cast<std::size_t>(g) * m_split;
        if (!is_root) {
            for (int cb = 0; cb < m_split; ++cb) {
                acc += coeff[cb] *
                       DotCodewordPair(sizes_one, dot_offsets_one, D_one,
                                       cb, static_cast<int>(code[cb]),
                                       target_book_one, target_code_one);
            }
            const int p_local_1b = ParentLocal1b(cluster, parent_override_1b, cur);
#ifndef NDEBUG
            if (p_local_1b <= 0) {
                throw std::runtime_error("DotReconstructionWithOneCodeword_TwoBookVirtual: non-root missing parent.");
            }
#endif
            cur = p_local_1b - 1;
        } else {
            for (int cb = 0; cb < m_split; ++cb) {
                acc += coeff[cb] *
                       DotCodewordPairCross(sizes_root, sizes_one, dot_offsets_root_one, D_root_one,
                                            cb, static_cast<int>(code[cb]),
                                            target_book_one, target_code_one);
            }
            break;
        }
    }
    return acc;
}

template <typename CodeT>
void ComputeLinkagedNormsTwoBookFromLinkage(const BaseEncoding& base,
                                         const LinkageStructure& linkage,
                                         const std::vector<CodeT>& codes,
                                         const CodebookMeta& meta_root,
                                         const CodebookMeta& meta_one,
                                         std::vector<float>* norms_out) {
    const int n_base = base.B.cols;
    const int m = base.B.rows;
    norms_out->assign(static_cast<std::size_t>(n_base), 0.0f);
    if (n_base <= 0 || m <= 0) {
        return;
    }

    if (meta_root.m != m || meta_one.m != m || meta_root.d != meta_one.d) {
        return;
    }

    const std::vector<int> dot_offsets_root = BuildDotOffsets(meta_root.sizes);
    const std::vector<int> dot_offsets_one = BuildDotOffsets(meta_one.sizes);
    const std::vector<int> dot_offsets_root_one = BuildCrossDotOffsets(meta_root.sizes, meta_one.sizes);
    std::vector<float> D_root;
    std::vector<float> D_one;
    std::vector<float> D_root_one;
    {
        BlasSetThreads(std::max(1, omp_get_max_threads()));
        PrecomputeCodebookDots(meta_root, dot_offsets_root, &D_root);
        PrecomputeCodebookDots(meta_one, dot_offsets_one, &D_one);
        PrecomputeCodebookDotsCross(meta_root, meta_one, dot_offsets_root_one, &D_root_one);
        BlasSetThreads(1);
    }

    const float* __restrict a = base.a.data.data();

    for (const auto& cluster : linkage.clusters) {
        const int csize = static_cast<int>(cluster.indices.size());
        if (csize <= 0) {
            continue;
        }

        const int cluster_max_depth = static_cast<int>(cluster.depth_offsets.size()) - 2;
        // Depth 0: root nodes (use root codebooks).
        {
            const int begin0 = cluster.depth_offsets[0];
            const int end0 = cluster.depth_offsets[1];
            for (int pos = begin0; pos < end0; ++pos) {
                const int g = cluster.indices[static_cast<std::size_t>(pos)];
                const CodeT* __restrict code = codes.data() + static_cast<std::size_t>(g) * m;
                const float* __restrict coeff = a + static_cast<std::size_t>(g) * m;

                float norm = 0.0f;
                for (int i = 0; i < m; ++i) {
                    const float li = coeff[i];
                    const int ci = static_cast<int>(code[i]);
                    norm += li * li *
                            DotCodewordPair(meta_root.sizes, dot_offsets_root, D_root.data(), i, ci, i, ci);
                    for (int j = i + 1; j < m; ++j) {
                        const float lj = coeff[j];
                        const int cj = static_cast<int>(code[j]);
                        norm += 2.0f * li * lj *
                                DotCodewordPair(meta_root.sizes, dot_offsets_root, D_root.data(), i, ci, j, cj);
                    }
                }
                (*norms_out)[static_cast<std::size_t>(g)] = norm;
            }
        }

        // Depth >= 1: residual nodes (use one codebooks).
        for (int dep = 1; dep <= cluster_max_depth; ++dep) {
            const int begin = cluster.depth_offsets[static_cast<std::size_t>(dep)];
            const int end = cluster.depth_offsets[static_cast<std::size_t>(dep + 1)];
            for (int pos = begin; pos < end; ++pos) {
                const int g = cluster.indices[static_cast<std::size_t>(pos)];
                const CodeT* __restrict code = codes.data() + static_cast<std::size_t>(g) * m;
                const float* __restrict coeff = a + static_cast<std::size_t>(g) * m;

                const int p_local_1b = cluster.parent_local[static_cast<std::size_t>(pos)];
#ifndef NDEBUG
                if (p_local_1b <= 0 || p_local_1b > csize) {
                    throw std::runtime_error("ComputeLinkagedNormsTwoBookFromLinkage: invalid parent for dep>=1 node.");
                }
#endif
                const int p_local = static_cast<int>(p_local_1b) - 1;
                const int gp = cluster.indices[static_cast<std::size_t>(p_local)];
                const float parent_norm = (*norms_out)[static_cast<std::size_t>(gp)];

                float residual_norm = 0.0f;
                for (int i = 0; i < m; ++i) {
                    const float li = coeff[i];
                    const int ci = static_cast<int>(code[i]);
                    residual_norm += li * li *
                                     DotCodewordPair(meta_one.sizes, dot_offsets_one, D_one.data(), i, ci, i, ci);
                    for (int j = i + 1; j < m; ++j) {
                        const float lj = coeff[j];
                        const int cj = static_cast<int>(code[j]);
                        residual_norm += 2.0f * li * lj *
                                         DotCodewordPair(meta_one.sizes, dot_offsets_one, D_one.data(), i, ci, j, cj);
                    }
                }

                float cross = 0.0f;
                for (int cb = 0; cb < m; ++cb) {
                    const float li = coeff[cb];
                    if (li == 0.0f) {
                        continue;
                    }
                    const int target_code = static_cast<int>(code[cb]);
                    const float dotp = DotReconstructionWithOneCodeword_TwoBook<CodeT>(
                        cluster, p_local, nullptr,
                        codes.data(), a, m,
                        meta_root.sizes, dot_offsets_root_one, D_root_one.data(),
                        meta_one.sizes, dot_offsets_one, D_one.data(),
                        cb, target_code);
                    cross += 2.0f * li * dotp;
                }

                (*norms_out)[static_cast<std::size_t>(g)] = parent_norm + residual_norm + cross;
            }
        }
    }
}

template <typename CodeT>
void ComputeLinkagedNormsTwoBookFromLinkageVirtual(const BaseEncoding& base_real,
                                                const VirtualEncoding& virt,
                                                const LinkageStructure& linkage,
                                                const CodebookMeta& meta_root,
                                                const CodebookMeta& meta_one,
                                                int n_base_real,
                                                std::vector<float>* norms_out) {
    const int m = base_real.B.rows;
    const int n_total = n_base_real + virt.n_virtual();
    norms_out->assign(static_cast<std::size_t>(n_total), 0.0f);
    if (n_total <= 0 || m <= 0) {
        return;
    }
    if (base_real.a.rows != m || virt.m != m) {
        return;
    }

    if (meta_root.m != m || meta_one.m != m || meta_root.d != meta_one.d) {
        return;
    }

    const std::vector<int> dot_offsets_root = BuildDotOffsets(meta_root.sizes);
    const std::vector<int> dot_offsets_one = BuildDotOffsets(meta_one.sizes);
    const std::vector<int> dot_offsets_root_one = BuildCrossDotOffsets(meta_root.sizes, meta_one.sizes);

    std::vector<float> D_root;
    std::vector<float> D_one;
    std::vector<float> D_root_one;
    {
        BlasSetThreads(std::max(1, omp_get_max_threads()));
        PrecomputeCodebookDots(meta_root, dot_offsets_root, &D_root);
        PrecomputeCodebookDots(meta_one, dot_offsets_one, &D_one);
        PrecomputeCodebookDotsCross(meta_root, meta_one, dot_offsets_root_one, &D_root_one);
        BlasSetThreads(1);
    }

    const float* __restrict a_real = base_real.a.data.data();

    for (const auto& cluster : linkage.clusters) {
        const int cid = cluster.cluster_id;
        const int n_virt = std::max(0, cluster.n_virtual);
        const int n_real = static_cast<int>(cluster.indices.size());
        const int n_total_local = n_virt + n_real;
        if (n_total_local <= 0) {
            continue;
        }
        if (static_cast<int>(cluster.parent_local.size()) != n_total_local) {
            LogWarn("ComputeLinkagedNorms[cid=" + std::to_string(cid) + "]: parent_local.size()=" +
                    std::to_string(cluster.parent_local.size()) + " != n_total_local=" +
                    std::to_string(n_total_local) + "; linkage structure corrupt, skipping.");
            continue;
        }
        std::vector<float> norm_local(static_cast<std::size_t>(n_total_local), 0.0f);

        const int cluster_max_depth = static_cast<int>(cluster.depth_offsets.size()) - 2;
        // Virtual roots (dense prefix): always depth 0 and use root codebooks.
        {
            for (int v_local = 0; v_local < n_virt; ++v_local) {
                const FullCode* __restrict code = virt.B_by_cluster[static_cast<std::size_t>(cid)].Col(v_local);
                const float* __restrict coeff = virt.a_by_cluster[static_cast<std::size_t>(cid)].Col(v_local);

                float norm = 0.0f;
                for (int i = 0; i < m; ++i) {
                    const float li = coeff[i];
                    const int ci = static_cast<int>(code[i]);
                    norm += li * li *
                            DotCodewordPair(meta_root.sizes, dot_offsets_root, D_root.data(), i, ci, i, ci);
                    for (int j = i + 1; j < m; ++j) {
                        const float lj = coeff[j];
                        const int cj = static_cast<int>(code[j]);
                        norm += 2.0f * li * lj *
                                DotCodewordPair(meta_root.sizes, dot_offsets_root, D_root.data(), i, ci, j, cj);
                    }
                }
                norm_local[static_cast<std::size_t>(v_local)] = norm;
            }
        }

        // Depth 0 real roots: use root codebooks.
        {
            const int begin0 = cluster.depth_offsets.empty() ? 0 : cluster.depth_offsets[0];
            const int end0 = (cluster.depth_offsets.size() >= 2) ? cluster.depth_offsets[1] : 0;
            for (int real_idx = begin0; real_idx < end0; ++real_idx) {
                const int g = cluster.indices[static_cast<std::size_t>(real_idx)];
                const FullCode* __restrict code = base_real.B.Col(g);
                const float* __restrict coeff = a_real + static_cast<std::size_t>(g) * m;

                float norm = 0.0f;
                for (int i = 0; i < m; ++i) {
                    const float li = coeff[i];
                    const int ci = static_cast<int>(code[i]);
                    norm += li * li *
                            DotCodewordPair(meta_root.sizes, dot_offsets_root, D_root.data(), i, ci, i, ci);
                    for (int j = i + 1; j < m; ++j) {
                        const float lj = coeff[j];
                        const int cj = static_cast<int>(code[j]);
                        norm += 2.0f * li * lj *
                                DotCodewordPair(meta_root.sizes, dot_offsets_root, D_root.data(), i, ci, j, cj);
                    }
                }
                const int local = n_virt + real_idx;
                norm_local[static_cast<std::size_t>(local)] = norm;
                (*norms_out)[static_cast<std::size_t>(g)] = norm;
            }
        }

        // Depth >= 1 residual nodes: use one codebooks; parent can be real or virtual.
        for (int dep = 1; dep <= cluster_max_depth; ++dep) {
            const int begin = cluster.depth_offsets[static_cast<std::size_t>(dep)];
            const int end = cluster.depth_offsets[static_cast<std::size_t>(dep + 1)];
            for (int real_idx = begin; real_idx < end; ++real_idx) {
                const int g = cluster.indices[static_cast<std::size_t>(real_idx)];
                const FullCode* __restrict code = base_real.B.Col(g);
                const float* __restrict coeff = a_real + static_cast<std::size_t>(g) * m;

                const int local = n_virt + real_idx;
                const int p_local_1b = ParentLocal1b(cluster, nullptr, local);
                const int p_local = static_cast<int>(p_local_1b) - 1;
                if (p_local < 0 || p_local >= n_total_local) {
                    LogWarn("ComputeLinkagedNorms[cid=" + std::to_string(cid) + " dep=" + std::to_string(dep) +
                            " real_idx=" + std::to_string(real_idx) + "]: p_local=" + std::to_string(p_local) +
                            " out of range [0," + std::to_string(n_total_local) + "); linkage data corrupt.");
                    continue;
                }
                const float parent_norm = norm_local[static_cast<std::size_t>(p_local)];

                float residual_norm = 0.0f;
                for (int i = 0; i < m; ++i) {
                    const float li = coeff[i];
                    const int ci = static_cast<int>(code[i]);
                    residual_norm += li * li *
                                     DotCodewordPair(meta_one.sizes, dot_offsets_one, D_one.data(), i, ci, i, ci);
                    for (int j = i + 1; j < m; ++j) {
                        const float lj = coeff[j];
                        const int cj = static_cast<int>(code[j]);
                        residual_norm += 2.0f * li * lj *
                                         DotCodewordPair(meta_one.sizes, dot_offsets_one, D_one.data(), i, ci, j, cj);
                    }
                }

                float cross = 0.0f;
                for (int cb = 0; cb < m; ++cb) {
                    const float li = coeff[cb];
                    if (li == 0.0f) {
                        continue;
                    }
                    const int target_book = cb;
                    const int target_code = static_cast<int>(code[cb]);
                    const float dotp = DotReconstructionWithOneCodeword_TwoBookVirtual<CodeT>(
                        cluster, p_local, nullptr, base_real, virt,
                        a_real, m,
                        meta_root.sizes, dot_offsets_root_one, D_root_one.data(),
                        meta_one.sizes, dot_offsets_one, D_one.data(),
                        target_book, target_code);
                    cross += 2.0f * li * dotp;
                }

                const float total = parent_norm + residual_norm + cross;
                norm_local[static_cast<std::size_t>(local)] = total;
                (*norms_out)[static_cast<std::size_t>(g)] = total;
            }
        }
    }
}

void Kmeans1DQuantize(const float* data,
                      int n,
                      int k,
                      int max_iter,
                      std::vector<int>* assignments,
                      std::vector<float>* centers) {
    assignments->assign(static_cast<std::size_t>(n), 0);
    centers->assign(static_cast<std::size_t>(k), 0.0f);
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
        const int idx = (static_cast<long long>(i) * n) / k;
        (*centers)[static_cast<std::size_t>(i)] = sorted[static_cast<std::size_t>(std::min(idx, n - 1))];
    }

    const int threads = std::max(1, omp_get_max_threads());
    std::vector<int> counts(static_cast<std::size_t>(threads) * static_cast<std::size_t>(k), 0);
    std::vector<double> sums(static_cast<std::size_t>(threads) * static_cast<std::size_t>(k), 0.0);

    for (int it = 0; it < max_iter; ++it) {
        std::fill(counts.begin(), counts.end(), 0);
        std::fill(sums.begin(), sums.end(), 0.0);

        #pragma omp parallel default(none) shared(data, assignments, centers, counts, sums) firstprivate(n, k, threads)
        {
            const int tid = omp_get_thread_num();
            int* __restrict c_local = counts.data() + static_cast<std::size_t>(tid) * k;
            double* __restrict s_local = sums.data() + static_cast<std::size_t>(tid) * k;

            #pragma omp for schedule(static)
            for (int i = 0; i < n; ++i) {
                const float x = data[i];
                float best = std::fabs(x - (*centers)[0]);
                int best_k = 0;
                // Ternary (cmov / blend) pattern lets the auto-vectorizer emit
                // a SIMD select instead of a conditional branch per iteration.
                for (int j = 1; j < k; ++j) {
                    const float dist = std::fabs(x - (*centers)[static_cast<std::size_t>(j)]);
                    const bool closer = (dist < best);
                    best   = closer ? dist : best;
                    best_k = closer ? j    : best_k;
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
                cnt += counts[static_cast<std::size_t>(t) * k + j];
                sum += sums[static_cast<std::size_t>(t) * k + j];
            }
            if (cnt <= 0) {
                continue;
            }
            const auto newc = static_cast<float>(sum / static_cast<double>(cnt));
            if (std::fabs(newc - (*centers)[static_cast<std::size_t>(j)]) > 1e-6f) {
                converged = false;
            }
            (*centers)[static_cast<std::size_t>(j)] = newc;
        }
        if (converged) {
            break;
        }
    }
}

LinkageStructure BuildLinkageStructureFromListsForEval(std::vector<std::vector<int>> lists,
                                                   const std::vector<int>& parent) {
    LinkageStructure out;
    const int nlist = static_cast<int>(lists.size());
    if (nlist <= 0) {
        return out;
    }

    out.clusters.reserve(static_cast<std::size_t>(nlist));
    int max_depth_global = 0;
    for (int cid = 0; cid < nlist; ++cid) {
        auto indices = std::move(lists[static_cast<std::size_t>(cid)]);
        const int csize = static_cast<int>(indices.size());
        if (csize == 0) {
            continue;
        }

        LinkageStructure::Cluster cluster;
        cluster.cluster_id = cid;
        cluster.n_real = csize;
        cluster.n_virtual = 0;
        cluster.indices = std::move(indices);
        cluster.parent_local.assign(static_cast<std::size_t>(csize), 0);

        std::unordered_map<int, int> global_to_local;
        global_to_local.reserve(static_cast<std::size_t>(csize) * 2);
        for (int local = 0; local < csize; ++local) {
            global_to_local[cluster.indices[static_cast<std::size_t>(local)]] = local;
        }

        for (int local = 0; local < csize; ++local) {
            const int g = cluster.indices[static_cast<std::size_t>(local)];
            const int p = parent[static_cast<std::size_t>(g)];
            if (p == 0) {
                cluster.parent_local[static_cast<std::size_t>(local)] = 0;
                continue;
            }
            const int gparent = p - 1;
            auto it = global_to_local.find(gparent);
            if (it != global_to_local.end()) {
                cluster.parent_local[static_cast<std::size_t>(local)] = it->second + 1;
            }
        }

        FinalizeLinkageClusterDepthOrderInPlace(&cluster);
        max_depth_global = std::max(max_depth_global, static_cast<int>(cluster.depth_offsets.size()) - 2);
        out.clusters.push_back(std::move(cluster));
    }

    out.max_depth = max_depth_global;
    return out;
}

LinkageStructure BuildLinkageStructureFromParentForEval(const BaseEncoding& base,
                                                    const std::vector<int>& parent,
                                                    int nlist) {
    LinkageStructure out;
    const int n_base = base.B.cols;
    if (n_base <= 0 || nlist <= 0 || static_cast<int>(parent.size()) != n_base) {
        return out;
    }

    std::vector<std::vector<int>> lists(static_cast<std::size_t>(nlist));
    // Cluster id is defined by the root node's C_root layer-0 code. For depth>0 nodes, B(0,i)
    // belongs to C_one and is NOT the cluster id, so we infer it by walking to the root.
    std::vector<int> root_idx(static_cast<std::size_t>(n_base), -1);
    std::vector<int> stack;
    stack.reserve(64);
    for (int i = 0; i < n_base; ++i) {
        if (root_idx[static_cast<std::size_t>(i)] >= 0) {
            continue;
        }
        int v = i;
        stack.clear();
        while (true) {
            const int rv = root_idx[static_cast<std::size_t>(v)];
            if (rv >= 0) {
                break;
            }
            const int p = parent[static_cast<std::size_t>(v)];
            if (p == 0) {
                root_idx[static_cast<std::size_t>(v)] = v;
                break;
            }
            stack.push_back(v);
            v = p - 1;
        }
        const int root = root_idx[static_cast<std::size_t>(v)];
        for (int s = static_cast<int>(stack.size()) - 1; s >= 0; --s) {
            const int u = stack[static_cast<std::size_t>(s)];
            root_idx[static_cast<std::size_t>(u)] = root;
        }
    }

    for (int i = 0; i < n_base; ++i) {
        const int r = root_idx[static_cast<std::size_t>(i)];
        // if (r < 0 || r >= n_base) continue;
        const int cid = static_cast<int>(base.B(0, r));
        // if (cid < 0 || cid >= nlist) continue;
        lists[static_cast<std::size_t>(cid)].push_back(i);
    }

    return BuildLinkageStructureFromListsForEval(std::move(lists), parent);
}

LinkageStructure BuildLinkageStructureFromParentAndClusterIdForEval(const std::vector<int>& parent,
                                                                const std::vector<int>& cluster_id,
                                                                int nlist) {
    LinkageStructure out;
    const int n_base = static_cast<int>(cluster_id.size());
    if (n_base <= 0 || nlist <= 0 || static_cast<int>(parent.size()) != n_base) {
        return out;
    }

    std::vector<std::vector<int>> lists(static_cast<std::size_t>(nlist));
    for (int i = 0; i < n_base; ++i) {
        const int cid = cluster_id[static_cast<std::size_t>(i)];
        // if (cid < 0 || cid >= nlist) continue;
        lists[static_cast<std::size_t>(cid)].push_back(i);
    }
    return BuildLinkageStructureFromListsForEval(std::move(lists), parent);
}

LinkageStructure BuildLinkageStructureFromParentAndClusterIdForEvalVirtual(const std::vector<int>& parent,
                                                                       const std::vector<int>& cluster_id,
                                                                       int nlist,
                                                                       int n_base_real) {
    LinkageStructure out;
    const int n_total = static_cast<int>(cluster_id.size());
    if (n_total <= 0 || nlist <= 0 || static_cast<int>(parent.size()) != n_total) {
        return out;
    }
    if (n_base_real <= 0) {
        n_base_real = n_total;
    }

    std::vector<int> counts(static_cast<std::size_t>(nlist), 0);
    for (int g = 0; g < n_total; ++g) {
        const int cid = cluster_id[static_cast<std::size_t>(g)];
        // if (cid < 0 || cid >= nlist) continue;
        counts[static_cast<std::size_t>(cid)] += 1;
    }

    std::vector<std::vector<int>> lists(static_cast<std::size_t>(nlist));
    for (int cid = 0; cid < nlist; ++cid) {
        const int cnt = counts[static_cast<std::size_t>(cid)];
        if (cnt > 0) {
            lists[static_cast<std::size_t>(cid)].reserve(static_cast<std::size_t>(cnt));
        }
    }
    for (int g = 0; g < n_total; ++g) {
        const int cid = cluster_id[static_cast<std::size_t>(g)];
        lists[static_cast<std::size_t>(cid)].push_back(g);
    }

    out.clusters.reserve(static_cast<std::size_t>(nlist));
    int max_depth_global = 0;
    for (int cid = 0; cid < nlist; ++cid) {
        auto nodes = std::move(lists[static_cast<std::size_t>(cid)]);
        if (nodes.empty()) {
            continue;
        }

        LinkageStructure::Cluster cluster;
        cluster.cluster_id = cid;
        std::vector<int> real_ids;
        std::vector<int> virt_ids;
        real_ids.reserve(nodes.size());
        virt_ids.reserve(nodes.size());
        for (int g : nodes) {
            if (g < n_base_real) {
                real_ids.push_back(g);
            } else {
                virt_ids.push_back(g);
            }
        }
        std::sort(real_ids.begin(), real_ids.end());
        std::sort(virt_ids.begin(), virt_ids.end());

        cluster.n_real = static_cast<int>(real_ids.size());
        cluster.n_virtual = static_cast<int>(virt_ids.size());
        cluster.indices = std::move(real_ids);

        const int n_virt = cluster.n_virtual;
        const int n_real = cluster.n_real;
        const int n_total_local = n_virt + n_real;
        cluster.parent_local.assign(static_cast<std::size_t>(n_total_local), 0);

        // parent_local[v_local] == 0 for all virtual roots by construction.
        for (int real_idx = 0; real_idx < n_real; ++real_idx) {
            const int g = cluster.indices[static_cast<std::size_t>(real_idx)];
            const int p = parent[static_cast<std::size_t>(g)];
            const int local = n_virt + real_idx;
            if (p == 0) {
                cluster.parent_local[static_cast<std::size_t>(local)] = 0;
                continue;
            }
            const int gparent = p - 1;
            if (gparent < n_base_real) {
                const auto it = std::lower_bound(cluster.indices.begin(), cluster.indices.end(), gparent);
                if (it != cluster.indices.end() && *it == gparent) {
                    const int p_real = static_cast<int>(it - cluster.indices.begin());
                    cluster.parent_local[static_cast<std::size_t>(local)] = n_virt + p_real + 1;
                }
            } else {
                const auto itv = std::lower_bound(virt_ids.begin(), virt_ids.end(), gparent);
                if (itv != virt_ids.end() && *itv == gparent) {
                    const int v_local = static_cast<int>(itv - virt_ids.begin());
                    cluster.parent_local[static_cast<std::size_t>(local)] = v_local + 1;
                }
            }
        }

        FinalizeLinkageClusterDepthOrderInPlace(&cluster);
        max_depth_global = std::max(max_depth_global, static_cast<int>(cluster.depth_offsets.size()) - 2);
        out.clusters.push_back(std::move(cluster));
    }

    out.max_depth = max_depth_global;
    return out;
}

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
        // Use a reduction instead of early-exit break so the inner loop is
        // auto-vectorizable (SIMD OR-reduction over the k result entries).
        int found = 0;
#pragma omp simd reduction(|:found)
        for (int j = 0; j < k; ++j) {
            found |= (row[j] == g ? 1 : 0);
        }
        hits += found;
    }
    return static_cast<float>(hits) / static_cast<float>(nquery);
}

// Deterministic ordering: (dist, id). This avoids nondeterministic tie handling
// when many candidates share identical distances.
struct ByDistThenIdLess {
    bool operator()(const std::pair<float, int32_t>& a,
                    const std::pair<float, int32_t>& b) const {
        if (a.first != b.first) {
            return a.first < b.first;
        }
        return a.second < b.second;
    }
};

}  // namespace

namespace {

template <typename CodeT, bool UseParentLouds>
bool EvaluateRecallLinkageIvfImpl(const Config& config,
                                const Dataset& dataset,
                                const BaseEncoding& base,
                                const LinkageStructure& linkage,
                                const CodebookMeta& meta_root,
                                const CodebookMeta& meta_one,
                                int nlist,
                                int n_base_real,
                                int k,
                                RecallResult* result,
                                std::string* error) {
    const int n_queries = dataset.Xq.cols;
    const int m = base.B.rows;

    if (nlist <= 0) {
        if (error) {
            *error = "EvaluateRecallLinkage: invalid nlist.";
        }
        return false;
    }

    const int nprobe_cfg = std::max(1, config.eval.linkage_nprobe);
    const int nprobe = std::min(nprobe_cfg, nlist);

    const int h_norms = config.base.encode.hnorms;
    const int max_iter = 60;

    Timer norm_timer;
    std::vector<float> dbnorms(static_cast<std::size_t>(n_base_real), 0.0f);

    std::vector<CodeT> codes;
    if (!ConvertCodes(base.B, &codes, error)) {
        return false;
    }

    std::vector<float> norms;
    ComputeLinkagedNormsTwoBookFromLinkage(base, linkage, codes, meta_root, meta_one, &norms);

    std::vector<int> assignments;
    std::vector<float> centers;
    Kmeans1DQuantize(norms.data(), n_base_real, h_norms, max_iter, &assignments, &centers);

    for (int i = 0; i < n_base_real; ++i) {
        int idx = assignments[static_cast<std::size_t>(i)];
        // if (idx < 0 || idx >= h_norms) {
        //     idx = 0;
        // }
        dbnorms[static_cast<std::size_t>(i)] = centers[idx];
    }
    const double norm_seconds = norm_timer.ElapsedSeconds();

    std::vector<int> cluster_lookup(static_cast<std::size_t>(nlist), -1);
    for (int ci = 0; ci < static_cast<int>(linkage.clusters.size()); ++ci) {
        const int id = static_cast<int>(linkage.clusters[static_cast<std::size_t>(ci)].cluster_id);
        // if (id >= 0 && id < nlist) {
        cluster_lookup[static_cast<std::size_t>(id)] = ci;
        // }
    }

    std::vector<std::vector<int>> parent_by_cluster_override;
    if constexpr (UseParentLouds) {
        Timer t;
        parent_by_cluster_override.resize(linkage.clusters.size());
        const std::uint32_t select_stride =
            static_cast<std::uint32_t>(std::max(1, config.eval.parent_louds_select_stride));
        const std::uint32_t rank_log2 =
            static_cast<std::uint32_t>(std::max(1, std::min(10, config.eval.parent_louds_rank_words_per_super_log2)));
        std::vector<std::uint32_t> tmp_parent;
        std::vector<std::uint32_t> decoded_parent;
        stlq::succinct::ParentLOUDS louds;
        for (std::size_t ci = 0; ci < linkage.clusters.size(); ++ci) {
            const auto& cluster = linkage.clusters[ci];
            const std::size_t n_total = cluster.parent_local.size();
            if (n_total == 0) {
                continue;
            }
            tmp_parent.assign(n_total, 0u);
            for (std::size_t i = 0; i < n_total; ++i) {
                const int p = cluster.parent_local[i];
#ifndef NDEBUG
                if (p < 0 || p > static_cast<int>(n_total)) {
                    throw std::runtime_error("EvaluateRecallLinkage: invalid parent_local value for LOUDS build.");
                }
#endif
                tmp_parent[i] = static_cast<std::uint32_t>(p);
            }
            louds.BuildFromParent1Based(tmp_parent.data(),
                                        tmp_parent.size(),
                                        select_stride,
                                        rank_log2,
                                        /*build_indices=*/false);
            louds.DecodeParent1Based(&decoded_parent, n_total);
            auto& dst = parent_by_cluster_override[ci];
            dst.resize(n_total);
            for (std::size_t i = 0; i < n_total; ++i) {
                dst[i] = static_cast<int>(decoded_parent[i]);
            }
        }
        if (config.large.profile_timing) {
            LogInfo("Non-large linkage recall parent LOUDS build+decode time: " + std::to_string(t.ElapsedSeconds()) + "s");
        }
    }

    ColMajorMatrix<float> xCq_root(meta_root.total_cols, n_queries);
    ColMajorMatrix<float> xCq_one(meta_one.total_cols, n_queries);
    {
        BlasSetThreads(std::max(1, omp_get_max_threads()));
        Timer gemm_timer;
        Gemm(true, false, 1.0f, meta_root.flat, dataset.Xq, 0.0f, &xCq_root);
        Gemm(true, false, 1.0f, meta_one.flat, dataset.Xq, 0.0f, &xCq_one);
        BlasSetThreads(1);
        LogInfo("Linkage IVF ADC query GEMM time: " + std::to_string(gemm_timer.ElapsedSeconds()) + "s");
    }

    const std::vector<float> ivf_inv_norm = ComputeIvfInvNormFromFirstBook(meta_root, nlist);

    result->dists = ColMajorMatrix<float>(k, n_queries);
    result->indices = ColMajorMatrix<int>(k, n_queries);

    Timer scan_timer;
    const int* __restrict offsets_root = meta_root.offsets.data();
    const int* __restrict offsets_one = meta_one.offsets.data();

    #pragma omp parallel default(none) shared(xCq_root, xCq_one, cluster_lookup, linkage, parent_by_cluster_override, codes, base, dbnorms, offsets_root, offsets_one, result, ivf_inv_norm) firstprivate(n_queries, nlist, nprobe, n_base_real, m, k)
    {
        std::vector<float> dot_local;
        std::vector<float> probe_scores(static_cast<std::size_t>(nlist),
                                        -std::numeric_limits<float>::infinity());
        std::vector<std::pair<float, int32_t>> heap;
        std::vector<int> best_ids(static_cast<std::size_t>(nprobe), 0);
        std::vector<float> best_scores(static_cast<std::size_t>(nprobe),
                                       -std::numeric_limits<float>::infinity());

        #pragma omp for schedule(static)
        for (int qi = 0; qi < n_queries; ++qi) {
            const float* __restrict xCq_root_col = xCq_root.Col(qi);
            const float* __restrict xCq_one_col = xCq_one.Col(qi);

            for (int cid = 0; cid < nlist; ++cid) {
                probe_scores[static_cast<std::size_t>(cid)] =
                    xCq_root_col[cid] * ivf_inv_norm[static_cast<std::size_t>(cid)];
            }

            const int nprobe_sel =
                ivf::SelectTopClustersByScore(probe_scores.data(), nlist, nprobe,
                                              best_ids.data(), best_scores.data());

            heap.clear();

            for (int pi = 0; pi < nprobe_sel; ++pi) {
                const int cid = best_ids[static_cast<std::size_t>(pi)];
                const int cidx = cluster_lookup[static_cast<std::size_t>(cid)];
                if (cidx < 0) {
                    continue;
                }
                const auto& cluster = linkage.clusters[static_cast<std::size_t>(cidx)];
                const int* parent_1b = nullptr;
                if constexpr (UseParentLouds) {
                    parent_1b = parent_by_cluster_override[static_cast<std::size_t>(cidx)].data();
                } else {
                    parent_1b = cluster.parent_local.data();
                }
                const int csize = static_cast<int>(cluster.indices.size());
#ifndef NDEBUG
                if (csize <= 0) {
                    throw std::runtime_error("EvaluateRecallLinkage: unexpected empty cluster.");
                }
                if (cluster.depth_offsets.size() < 2) {
                    throw std::runtime_error("EvaluateRecallLinkage: cluster.depth_offsets too small.");
                }
                if (static_cast<int>(cluster.parent_local.size()) != csize) {
                    throw std::runtime_error("EvaluateRecallLinkage: expected parent_local.size()==cluster size.");
                }
#endif

                dot_local.assign(static_cast<std::size_t>(csize), 0.0f);
                const int cluster_max_depth = static_cast<int>(cluster.depth_offsets.size()) - 2;
                // Depth 0: root nodes use root codebooks / xCq_root.
                {
                    const int begin0 = cluster.depth_offsets[0];
                    const int end0 = cluster.depth_offsets[1];
                    for (int pos = begin0; pos < end0; ++pos) {
                        const int g = cluster.indices[static_cast<std::size_t>(pos)];
                        const CodeT* __restrict code = codes.data() + static_cast<std::size_t>(g) * m;
                        const float* __restrict coeff = base.a.data.data() + static_cast<std::size_t>(g) * m;

                        float acc = 0.0f;
                        for (int cb = 0; cb < m; ++cb) {
                            const int flat = offsets_root[cb] + static_cast<int>(code[cb]);
                            acc += coeff[cb] * xCq_root_col[flat];
                        }
                        dot_local[static_cast<std::size_t>(pos)] = acc;
                    }
                }

                // Depth >= 1: residual nodes use one codebooks / xCq_one.
                for (int dep = 1; dep <= cluster_max_depth; ++dep) {
                    const int begin = cluster.depth_offsets[static_cast<std::size_t>(dep)];
                    const int end = cluster.depth_offsets[static_cast<std::size_t>(dep + 1)];
                    for (int pos = begin; pos < end; ++pos) {
                        const int g = cluster.indices[static_cast<std::size_t>(pos)];
                        const CodeT* __restrict code = codes.data() + static_cast<std::size_t>(g) * m;
                        const float* __restrict coeff = base.a.data.data() + static_cast<std::size_t>(g) * m;

                        float acc = 0.0f;
                        for (int cb = 0; cb < m; ++cb) {
                            const int flat = offsets_one[cb] + static_cast<int>(code[cb]);
                            acc += coeff[cb] * xCq_one_col[flat];
                        }
                        dot_local[static_cast<std::size_t>(pos)] = acc;
                    }
                }

                for (int dep = 1; dep <= cluster_max_depth; ++dep) {
                    const int begin = cluster.depth_offsets[static_cast<std::size_t>(dep)];
                    const int end = cluster.depth_offsets[static_cast<std::size_t>(dep + 1)];
                    for (int pos = begin; pos < end; ++pos) {
#ifndef NDEBUG
                        const int p1 = parent_1b[pos];
                        if (p1 <= 0 || p1 > csize) {
                            throw std::runtime_error("EvaluateRecallLinkage: invalid parent for dep>=1 node.");
                        }
                        const int p_local = p1 - 1;
                        if (p_local >= pos) {
                            throw std::runtime_error("EvaluateRecallLinkage: parent not earlier than child.");
                        }
#else
                        const int p_local = parent_1b[pos] - 1;
#endif
                        dot_local[static_cast<std::size_t>(pos)] += dot_local[static_cast<std::size_t>(p_local)];
                    }
                }

                const int n_real_local =
                    (cluster.n_real > 0) ? std::min(cluster.n_real, csize) : csize;
                for (int local = 0; local < n_real_local; ++local) {
                    const int g = cluster.indices[static_cast<std::size_t>(local)];
#ifndef NDEBUG
                    if (g >= n_base_real) {
                        throw std::runtime_error("EvaluateRecallLinkage: unexpected g>=n_base_real in real-only cluster.");
                    }
#endif
                    const float dist = dbnorms[static_cast<std::size_t>(g)] -
                                       2.0f * dot_local[static_cast<std::size_t>(local)];
                    const auto gi = static_cast<int32_t>(g);
                    if (static_cast<int>(heap.size()) < k) {
                        heap.emplace_back(dist, gi);
                        if (static_cast<int>(heap.size()) == k) {
                            std::make_heap(heap.begin(), heap.end(), ByDistThenIdLess{});
                        }
                    } else if (dist < heap.front().first) {
                        std::pop_heap(heap.begin(), heap.end(), ByDistThenIdLess{});
                        heap.back() = {dist, gi};
                        std::push_heap(heap.begin(), heap.end(), ByDistThenIdLess{});
                    }
                }
            }

            float* out_d = result->dists.Col(qi);
            int* out_i = result->indices.Col(qi);
            if (!heap.empty()) {
                std::sort_heap(heap.begin(), heap.end(), ByDistThenIdLess{});
            }
            const int got = static_cast<int>(heap.size());
            for (int j = 0; j < got; ++j) {
                out_d[j] = heap[static_cast<std::size_t>(j)].first;
                out_i[j] = static_cast<int>(heap[static_cast<std::size_t>(j)].second);
            }
            for (int j = got; j < k; ++j) {
                out_d[j] = std::numeric_limits<float>::infinity();
                out_i[j] = -1;
            }
        }
    }

    LogInfo("Linkage IVF ADC scan+topk time: " + std::to_string(scan_timer.ElapsedSeconds()) + "s");
    LogInfo("Linkage IVF ADC norm-quant time: " + std::to_string(norm_seconds) + "s");
    result->recall = ComputeRecallAtKMatrix(dataset.gt, result->indices);
    return true;
}

template <typename CodeT, bool UseParentLouds>
bool EvaluateRecallLinkageIvfImplVirtual(const Config& config,
                                       const Dataset& dataset,
                                       const BaseEncoding& base_real,
                                       const VirtualEncoding& virt,
                                       const LinkageStructure& linkage,
                                       const CodebookMeta& meta_root,
                                       const CodebookMeta& meta_one,
                                       int nlist,
                                       int n_base_real,
                                       int k,
                                       RecallResult* result,
                                       std::string* error) {
    const int n_queries = dataset.Xq.cols;
    const int m = base_real.B.rows;
    if (n_base_real <= 0) {
        n_base_real = base_real.B.cols;
    }
    const int n_total = n_base_real + virt.n_virtual();

    if (nlist <= 0) {
        if (error) {
            *error = "EvaluateRecallLinkageVirtual: invalid nlist.";
        }
        return false;
    }
    if (n_total <= 0) {
        if (error) {
            *error = "EvaluateRecallLinkageVirtual: empty base.";
        }
        return false;
    }

    const int nprobe_cfg = std::max(1, config.eval.linkage_nprobe);
    const int nprobe = std::min(nprobe_cfg, nlist);

    const int h_norms = config.base.encode.hnorms;
    const int max_iter = 60;

    std::vector<float> dbnorms(static_cast<std::size_t>(n_base_real), 0.0f);

    Timer norm_timer;
    std::vector<float> norms;
    ComputeLinkagedNormsTwoBookFromLinkageVirtual<CodeT>(base_real, virt, linkage, meta_root, meta_one,
                                                      n_base_real, &norms);

    std::vector<int> assignments;
    std::vector<float> centers;
    Kmeans1DQuantize(norms.data(), n_base_real, h_norms, max_iter, &assignments, &centers);

    for (int i = 0; i < n_base_real; ++i) {
        int idx = assignments[static_cast<std::size_t>(i)];
        dbnorms[static_cast<std::size_t>(i)] = centers[static_cast<std::size_t>(idx)];
    }
    const double norm_seconds = norm_timer.ElapsedSeconds();

    std::vector<int> cluster_lookup(static_cast<std::size_t>(nlist), -1);
    for (int ci = 0; ci < static_cast<int>(linkage.clusters.size()); ++ci) {
        const int id = static_cast<int>(linkage.clusters[static_cast<std::size_t>(ci)].cluster_id);
        cluster_lookup[static_cast<std::size_t>(id)] = ci;
    }

    std::vector<std::vector<int>> parent_by_cluster_override;
    if constexpr (UseParentLouds) {
        Timer t;
        parent_by_cluster_override.resize(linkage.clusters.size());
        const std::uint32_t select_stride =
            static_cast<std::uint32_t>(std::max(1, config.eval.parent_louds_select_stride));
        const std::uint32_t rank_log2 =
            static_cast<std::uint32_t>(std::max(1, std::min(10, config.eval.parent_louds_rank_words_per_super_log2)));
        std::vector<std::uint32_t> tmp_parent;
        std::vector<std::uint32_t> decoded_parent;
        stlq::succinct::ParentLOUDS louds;
        for (std::size_t ci = 0; ci < linkage.clusters.size(); ++ci) {
            const auto& cluster = linkage.clusters[ci];
            const std::size_t n_total = cluster.parent_local.size();
            if (n_total == 0) {
                continue;
            }
            tmp_parent.assign(n_total, 0u);
            for (std::size_t i = 0; i < n_total; ++i) {
                const int p = cluster.parent_local[i];
#ifndef NDEBUG
                if (p < 0 || p > static_cast<int>(n_total)) {
                    throw std::runtime_error("EvaluateRecallLinkageVirtual: invalid parent_local value for LOUDS build.");
                }
#endif
                tmp_parent[i] = static_cast<std::uint32_t>(p);
            }
            louds.BuildFromParent1Based(tmp_parent.data(),
                                        tmp_parent.size(),
                                        select_stride,
                                        rank_log2,
                                        /*build_indices=*/false);
            louds.DecodeParent1Based(&decoded_parent, n_total);
            auto& dst = parent_by_cluster_override[ci];
            dst.resize(n_total);
            for (std::size_t i = 0; i < n_total; ++i) {
                dst[i] = static_cast<int>(decoded_parent[i]);
            }
        }
        if (config.large.profile_timing) {
            LogInfo("Non-large linkage recall parent LOUDS build+decode time: " + std::to_string(t.ElapsedSeconds()) + "s");
        }
    }

    ColMajorMatrix<float> xCq_root(meta_root.total_cols, n_queries);
    ColMajorMatrix<float> xCq_one(meta_one.total_cols, n_queries);
    {
        BlasSetThreads(std::max(1, omp_get_max_threads()));
        Timer gemm_timer;
        Gemm(true, false, 1.0f, meta_root.flat, dataset.Xq, 0.0f, &xCq_root);
        Gemm(true, false, 1.0f, meta_one.flat, dataset.Xq, 0.0f, &xCq_one);
        BlasSetThreads(1);
        LogInfo("Linkage IVF ADC query GEMM time: " + std::to_string(gemm_timer.ElapsedSeconds()) + "s");
    }

    const std::vector<float> ivf_inv_norm = ComputeIvfInvNormFromFirstBook(meta_root, nlist);

    result->dists = ColMajorMatrix<float>(k, n_queries);
    result->indices = ColMajorMatrix<int>(k, n_queries);

    const int* __restrict offsets_root = meta_root.offsets.data();
    const int* __restrict offsets_one = meta_one.offsets.data();
    const float* __restrict a_real = base_real.a.data.data();

    Timer scan_timer;
    #pragma omp parallel default(none) shared(xCq_root, xCq_one, cluster_lookup, linkage, parent_by_cluster_override, base_real, virt, dbnorms, offsets_root, offsets_one, a_real, result, ivf_inv_norm) firstprivate(n_queries, nlist, nprobe, n_base_real, m, k)
    {
        std::vector<float> dot_local;
        std::vector<float> probe_scores(static_cast<std::size_t>(nlist),
                                        -std::numeric_limits<float>::infinity());
        std::vector<std::pair<float, int32_t>> heap;
        std::vector<int> best_ids(static_cast<std::size_t>(nprobe), 0);
        std::vector<float> best_scores(static_cast<std::size_t>(nprobe),
                                       -std::numeric_limits<float>::infinity());

        #pragma omp for schedule(static)
        for (int qi = 0; qi < n_queries; ++qi) {
            const float* __restrict xCq_root_col = xCq_root.Col(qi);
            const float* __restrict xCq_one_col = xCq_one.Col(qi);

            for (int cid = 0; cid < nlist; ++cid) {
                probe_scores[static_cast<std::size_t>(cid)] =
                    xCq_root_col[cid] * ivf_inv_norm[static_cast<std::size_t>(cid)];
            }

            const int nprobe_sel =
                ivf::SelectTopClustersByScore(probe_scores.data(), nlist, nprobe,
                                              best_ids.data(), best_scores.data());

            heap.clear();

            for (int pi = 0; pi < nprobe_sel; ++pi) {
                const int cid = best_ids[static_cast<std::size_t>(pi)];
                const int cidx = cluster_lookup[static_cast<std::size_t>(cid)];
                if (cidx < 0) {
                    continue;
                }
                const auto& cluster = linkage.clusters[static_cast<std::size_t>(cidx)];
                const int* parent_1b = nullptr;
                if constexpr (UseParentLouds) {
                    parent_1b = parent_by_cluster_override[static_cast<std::size_t>(cidx)].data();
                } else {
                    parent_1b = cluster.parent_local.data();
                }
                const int n_virt = std::max(0, cluster.n_virtual);
                const int n_real = static_cast<int>(cluster.indices.size());
                const int n_total_local = n_virt + n_real;
#ifndef NDEBUG
                if (n_total_local <= 0) {
                    throw std::runtime_error("EvaluateRecallLinkageVirtual: unexpected empty cluster.");
                }
                if (cluster.depth_offsets.size() < 2) {
                    throw std::runtime_error("EvaluateRecallLinkageVirtual: cluster.depth_offsets too small.");
                }
                if (static_cast<int>(cluster.parent_local.size()) != n_total_local) {
                    throw std::runtime_error("EvaluateRecallLinkageVirtual: expected parent_local.size()==n_total_local.");
                }
#endif

                dot_local.assign(static_cast<std::size_t>(n_total_local), 0.0f);

                const int cluster_max_depth = static_cast<int>(cluster.depth_offsets.size()) - 2;
                // Virtual roots (dense prefix): use root codebooks / xCq_root.
                for (int v_local = 0; v_local < n_virt; ++v_local) {
                    const FullCode* __restrict code =
                        virt.B_by_cluster[static_cast<std::size_t>(cid)].Col(v_local);
                    const float* __restrict coeff =
                        virt.a_by_cluster[static_cast<std::size_t>(cid)].Col(v_local);
                    float acc = 0.0f;
                    for (int cb = 0; cb < m; ++cb) {
                        const int flat = offsets_root[cb] + static_cast<int>(code[cb]);
                        acc += coeff[cb] * xCq_root_col[flat];
                    }
                    dot_local[static_cast<std::size_t>(v_local)] = acc;
                }

                // Depth 0 real roots (sparse): use root codebooks / xCq_root.
                {
                    const int begin0 = cluster.depth_offsets[0];
                    const int end0 = cluster.depth_offsets[1];
                    for (int real_idx = begin0; real_idx < end0; ++real_idx) {
                        const int g = cluster.indices[static_cast<std::size_t>(real_idx)];
                        const FullCode* __restrict code = base_real.B.Col(g);
                        const float* __restrict coeff = a_real + static_cast<std::size_t>(g) * m;

                        float acc = 0.0f;
                        for (int cb = 0; cb < m; ++cb) {
                            const int flat = offsets_root[cb] + static_cast<int>(code[cb]);
                            acc += coeff[cb] * xCq_root_col[flat];
                        }
                        const int local = n_virt + real_idx;
                        dot_local[static_cast<std::size_t>(local)] = acc;
                    }
                }

                // Depth >= 1 residual nodes: real only, use one codebooks / xCq_one.
                for (int dep = 1; dep <= cluster_max_depth; ++dep) {
                    const int begin = cluster.depth_offsets[static_cast<std::size_t>(dep)];
                    const int end = cluster.depth_offsets[static_cast<std::size_t>(dep + 1)];
                    for (int real_idx = begin; real_idx < end; ++real_idx) {
                        const int g = cluster.indices[static_cast<std::size_t>(real_idx)];
                        const FullCode* __restrict code = base_real.B.Col(g);
                        const float* __restrict coeff = a_real + static_cast<std::size_t>(g) * m;

                        float acc = 0.0f;
                        for (int cb = 0; cb < m; ++cb) {
                            const int flat = offsets_one[cb] + static_cast<int>(code[cb]);
                            acc += coeff[cb] * xCq_one_col[flat];
                        }
                        const int local = n_virt + real_idx;
                        dot_local[static_cast<std::size_t>(local)] = acc;
                    }
                }

                for (int dep = 1; dep <= cluster_max_depth; ++dep) {
                    const int begin = cluster.depth_offsets[static_cast<std::size_t>(dep)];
                    const int end = cluster.depth_offsets[static_cast<std::size_t>(dep + 1)];
                    for (int real_idx = begin; real_idx < end; ++real_idx) {
                        const int local = n_virt + real_idx;
#ifndef NDEBUG
                        const int p1 = parent_1b[local];
                        if (p1 <= 0 || p1 > n_total_local) {
                            throw std::runtime_error("EvaluateRecallLinkageVirtual: invalid parent for dep>=1 node.");
                        }
                        const int p_local = p1 - 1;
                        if (p_local >= local) {
                            throw std::runtime_error("EvaluateRecallLinkageVirtual: parent not earlier than child.");
                        }
#else
                        const int p_local = parent_1b[local] - 1;
#endif
                        dot_local[static_cast<std::size_t>(local)] += dot_local[static_cast<std::size_t>(p_local)];
                    }
                }

                for (int real_idx = 0; real_idx < n_real; ++real_idx) {
                    const int g = cluster.indices[static_cast<std::size_t>(real_idx)];
#ifndef NDEBUG
                    if (g >= n_base_real) {
                        throw std::runtime_error("EvaluateRecallLinkageVirtual: unexpected real g>=n_base_real.");
                    }
#endif
                    const int local = n_virt + real_idx;
                    const float dist = dbnorms[static_cast<std::size_t>(g)] -
                                       2.0f * dot_local[static_cast<std::size_t>(local)];
                    const auto gi = static_cast<int32_t>(g);
                    if (static_cast<int>(heap.size()) < k) {
                        heap.emplace_back(dist, gi);
                        if (static_cast<int>(heap.size()) == k) {
                            std::make_heap(heap.begin(), heap.end(), ByDistThenIdLess{});
                        }
                    } else if (dist < heap.front().first) {
                        std::pop_heap(heap.begin(), heap.end(), ByDistThenIdLess{});
                        heap.back() = {dist, gi};
                        std::push_heap(heap.begin(), heap.end(), ByDistThenIdLess{});
                    }
                }
            }

            float* out_d = result->dists.Col(qi);
            int* out_i = result->indices.Col(qi);
            if (!heap.empty()) {
                std::sort_heap(heap.begin(), heap.end(), ByDistThenIdLess{});
            }
            const int got = static_cast<int>(heap.size());
            for (int j = 0; j < got; ++j) {
                out_d[j] = heap[static_cast<std::size_t>(j)].first;
                out_i[j] = static_cast<int>(heap[static_cast<std::size_t>(j)].second);
            }
            for (int j = got; j < k; ++j) {
                out_d[j] = std::numeric_limits<float>::infinity();
                out_i[j] = -1;
            }
        }
    }
    LogInfo("Linkage IVF ADC scan+topk time: " + std::to_string(scan_timer.ElapsedSeconds()) + "s");
    LogInfo("Linkage IVF ADC norm-quant time: " + std::to_string(norm_seconds) + "s");
    return true;
}

}  // namespace

bool EvaluateRecallLinkage(const Config& config,
                         const Dataset& dataset,
                         const CodebookPack& C_root,
                         const CodebookPack& C_one,
                         const BaseEncoding& base,
                         const LinkageStructure& linkage,
                         int n_base_real,
                         RecallResult* result,
                         std::string* error) {
    if (!result) {
        return false;
    }
    const int n_base = base.B.cols;
    if (n_base_real <= 0 || n_base_real > n_base) {
        n_base_real = n_base;
    }

    int k = std::max(1, config.dataset.k);
    k = std::min(k, n_base_real);

    const int m = base.B.rows;
    CodebookMeta meta_root = BuildCodebookMeta(GatherBooks(C_root));
    CodebookMeta meta_one = BuildCodebookMeta(GatherBooks(C_one));
    if (meta_root.m != m || meta_one.m != m || meta_root.d != meta_one.d) {
        if (error) {
            *error = "Expected root/one codebooks to each match encoding rows.";
        }
        return false;
    }

    const int nlist = C_root.books.empty() ? 0 : C_root.books.front().cols;
    const int max_h = std::max(meta_root.max_h, meta_one.max_h);
    const bool use_parent_louds = config.eval.linkage_parent_louds_enable;
    if (max_h <= 256) {
        if (use_parent_louds) {
            return EvaluateRecallLinkageIvfImpl<uint8_t, true>(config, dataset, base, linkage, meta_root, meta_one,
                                                             nlist, n_base_real, k, result, error);
        }
        return EvaluateRecallLinkageIvfImpl<uint8_t, false>(config, dataset, base, linkage, meta_root, meta_one,
                                                          nlist, n_base_real, k, result, error);
    }
    if (use_parent_louds) {
        return EvaluateRecallLinkageIvfImpl<uint16_t, true>(config, dataset, base, linkage, meta_root, meta_one,
                                                          nlist, n_base_real, k, result, error);
    }
    return EvaluateRecallLinkageIvfImpl<uint16_t, false>(config, dataset, base, linkage, meta_root, meta_one,
                                                       nlist, n_base_real, k, result, error);
}

bool EvaluateRecallLinkageVirtual(const Config& config,
                                const Dataset& dataset,
                                const CodebookPack& C_root,
                                const CodebookPack& C_one,
                                const BaseEncoding& base_real,
                                const VirtualEncoding& virt,
                                const LinkageStructure& linkage,
                                int n_base_real,
                                RecallResult* result,
                                std::string* error) {
    if (!result) {
        return false;
    }
    if (n_base_real <= 0 || n_base_real > base_real.B.cols) {
        n_base_real = base_real.B.cols;
    }
    int k = std::max(1, config.dataset.k);
    k = std::min(k, n_base_real);

    const int m = base_real.B.rows;
    CodebookMeta meta_root = BuildCodebookMeta(GatherBooks(C_root));
    CodebookMeta meta_one = BuildCodebookMeta(GatherBooks(C_one));
    if (meta_root.m != m || meta_one.m != m || meta_root.d != meta_one.d) {
        if (error) {
            *error = "Expected root/one codebooks to each match encoding rows.";
        }
        return false;
    }

    const int nlist = C_root.books.empty() ? 0 : C_root.books.front().cols;
    const int max_h = std::max(meta_root.max_h, meta_one.max_h);
    const bool use_parent_louds = config.eval.linkage_parent_louds_enable;
    if (max_h <= 256) {
        if (use_parent_louds) {
            return EvaluateRecallLinkageIvfImplVirtual<uint8_t, true>(config, dataset, base_real, virt, linkage,
                                                                    meta_root, meta_one,
                                                                    nlist, n_base_real, k, result, error);
        }
        return EvaluateRecallLinkageIvfImplVirtual<uint8_t, false>(config, dataset, base_real, virt, linkage,
                                                                 meta_root, meta_one,
                                                                 nlist, n_base_real, k, result, error);
    }
    if (use_parent_louds) {
        return EvaluateRecallLinkageIvfImplVirtual<uint16_t, true>(config, dataset, base_real, virt, linkage,
                                                                 meta_root, meta_one,
                                                                 nlist, n_base_real, k, result, error);
    }
    return EvaluateRecallLinkageIvfImplVirtual<uint16_t, false>(config, dataset, base_real, virt, linkage,
                                                              meta_root, meta_one,
                                                              nlist, n_base_real, k, result, error);
}

bool EvaluateRecallLinkageVirtualFromParentWithClusterId(const Config& config,
                                                       const Dataset& dataset,
                                                       const CodebookPack& C_root,
                                                       const CodebookPack& C_one,
                                                       const BaseEncoding& base_real,
                                                       const VirtualEncoding& virt,
                                                       const std::vector<int>& parent,
                                                       const std::vector<int>& cluster_id,
                                                       int n_base_real,
                                                       RecallResult* result,
                                                       std::string* error) {
    if (!result) {
        return false;
    }
    if (n_base_real <= 0 || n_base_real > base_real.B.cols) {
        n_base_real = base_real.B.cols;
    }
    const int n_total = static_cast<int>(parent.size());
    if (static_cast<int>(cluster_id.size()) != n_total) {
        if (error) {
            *error = "Parent/cluster_id vector size does not match n_total.";
        }
        return false;
    }
    const int nlist = C_root.books.empty() ? 0 : C_root.books.front().cols;
    LinkageStructure linkage = BuildLinkageStructureFromParentAndClusterIdForEvalVirtual(parent, cluster_id, nlist, n_base_real);
    return EvaluateRecallLinkageVirtual(config, dataset, C_root, C_one, base_real, virt, linkage,
                                      n_base_real, result, error);
}

bool EvaluateRecallLinkageFromParent(const Config& config,
                                   const Dataset& dataset,
                                   const CodebookPack& C_root,
                                   const CodebookPack& C_one,
                                   const BaseEncoding& base,
                                   const std::vector<int>& parent,
                                   int n_base_real,
                                   RecallResult* result,
                                   std::string* error) {
    const int n_base = base.B.cols;
    const int m = base.B.rows;
    if (!result) {
        return false;
    }
    if (dataset.gt.size() != static_cast<std::size_t>(dataset.Xq.cols)) {
        if (error) {
            *error = "Ground truth size does not match query count.";
        }
        return false;
    }
    if (static_cast<int>(parent.size()) != n_base) {
        if (error) {
            *error = "Parent vector size does not match base size.";
        }
        return false;
    }

    if (n_base_real <= 0 || n_base_real > n_base) {
        n_base_real = n_base;
    }
    int k = std::max(1, config.dataset.k);
    k = std::min(k, n_base_real);

    CodebookMeta meta_root = BuildCodebookMeta(GatherBooks(C_root));
    CodebookMeta meta_one = BuildCodebookMeta(GatherBooks(C_one));
    if (meta_root.m != m || meta_one.m != m || meta_root.d != meta_one.d) {
        if (error) {
            *error = "Expected root/one codebooks to each match encoding rows.";
        }
        return false;
    }

    const int nlist = C_root.books.empty() ? 0 : C_root.books.front().cols;
    LinkageStructure linkage = BuildLinkageStructureFromParentForEval(base, parent, nlist);
    const int max_h = std::max(meta_root.max_h, meta_one.max_h);
    const bool use_parent_louds = config.eval.linkage_parent_louds_enable;
    if (max_h <= 256) {
        if (use_parent_louds) {
            return EvaluateRecallLinkageIvfImpl<uint8_t, true>(config, dataset, base, linkage, meta_root, meta_one,
                                                             nlist, n_base_real, k, result, error);
        }
        return EvaluateRecallLinkageIvfImpl<uint8_t, false>(config, dataset, base, linkage, meta_root, meta_one,
                                                          nlist, n_base_real, k, result, error);
    }
    if (use_parent_louds) {
        return EvaluateRecallLinkageIvfImpl<uint16_t, true>(config, dataset, base, linkage, meta_root, meta_one,
                                                          nlist, n_base_real, k, result, error);
    }
    return EvaluateRecallLinkageIvfImpl<uint16_t, false>(config, dataset, base, linkage, meta_root, meta_one,
                                                       nlist, n_base_real, k, result, error);
}

bool EvaluateRecallLinkageFromParentWithClusterId(const Config& config,
                                                const Dataset& dataset,
                                                const CodebookPack& C_root,
                                                const CodebookPack& C_one,
                                                const BaseEncoding& base,
                                                const std::vector<int>& parent,
                                                const std::vector<int>& cluster_id,
                                                int n_base_real,
                                                RecallResult* result,
                                                std::string* error) {
    const int n_base = base.B.cols;
    const int m = base.B.rows;
    if (!result) {
        return false;
    }
    if (dataset.gt.size() != static_cast<std::size_t>(dataset.Xq.cols)) {
        if (error) {
            *error = "Ground truth size does not match query count.";
        }
        return false;
    }
    if (static_cast<int>(parent.size()) != n_base || static_cast<int>(cluster_id.size()) != n_base) {
        if (error) {
            *error = "Parent/cluster_id vector size does not match base size.";
        }
        return false;
    }

    if (n_base_real <= 0 || n_base_real > n_base) {
        n_base_real = n_base;
    }
    int k = std::max(1, config.dataset.k);
    k = std::min(k, n_base_real);

    CodebookMeta meta_root = BuildCodebookMeta(GatherBooks(C_root));
    CodebookMeta meta_one = BuildCodebookMeta(GatherBooks(C_one));
    if (meta_root.m != m || meta_one.m != m || meta_root.d != meta_one.d) {
        if (error) {
            *error = "Expected root/one codebooks to each match encoding rows.";
        }
        return false;
    }

    const int nlist = C_root.books.empty() ? 0 : C_root.books.front().cols;
    LinkageStructure linkage = BuildLinkageStructureFromParentAndClusterIdForEval(parent, cluster_id, nlist);

    const int max_h = std::max(meta_root.max_h, meta_one.max_h);
    const bool use_parent_louds = config.eval.linkage_parent_louds_enable;
    if (max_h <= 256) {
        if (use_parent_louds) {
            return EvaluateRecallLinkageIvfImpl<uint8_t, true>(config, dataset, base, linkage, meta_root, meta_one,
                                                             nlist, n_base_real, k, result, error);
        }
        return EvaluateRecallLinkageIvfImpl<uint8_t, false>(config, dataset, base, linkage, meta_root, meta_one,
                                                          nlist, n_base_real, k, result, error);
    }
    if (use_parent_louds) {
        return EvaluateRecallLinkageIvfImpl<uint16_t, true>(config, dataset, base, linkage, meta_root, meta_one,
                                                          nlist, n_base_real, k, result, error);
    }
    return EvaluateRecallLinkageIvfImpl<uint16_t, false>(config, dataset, base, linkage, meta_root, meta_one,
                                                       nlist, n_base_real, k, result, error);
}

}  // namespace stlq
