#include "stlq/quantizer/precomp_large_root.h"

#include <algorithm>
#include <cmath>

#include "stlq/core/blas.h"
#include "stlq/core/kernel_provider_cpu.h"
#include "stlq/core/threading.h"
#include "stlq/quantizer/linear_algebra.h"

namespace stlq {

namespace {

inline float Norm2(const float* x, int d) {
    float s = 0.0f;
    #pragma omp simd reduction(+:s)
    for (int i = 0; i < d; ++i) {
        const float v = x[i];
        s += v * v;
    }
    return s;
}

}  // namespace

bool BuildPrecompLargeRoot(const CodebookPack& C_root,
                           StreamKernelProvider* kernels,
                           const PrecompLargeRootBuildOptions& opts,
                           PrecompLargeRoot* out,
                           std::string* err) {
    if (!out || C_root.books.empty()) {
        if (err) {
            *err = "BuildPrecompLargeRoot: empty input.";
        }
        return false;
    }

    CpuStreamKernels cpu_kernels;
    if (!kernels) {
        kernels = &cpu_kernels;
    }
    const int m = static_cast<int>(C_root.books.size());
    const int d = C_root.books.front().rows;
    if (m <= 0 || d <= 0) {
        if (err) {
            *err = "BuildPrecompLargeRoot: invalid codebooks.";
        }
        return false;
    }

    out->ready = false;
    out->d = d;
    out->m = m;
    out->h_vec.clear();
    out->h_vec.reserve(m);
    for (const auto& book : C_root.books) {
        if (book.rows != d) {
            if (err) {
                *err = "BuildPrecompLargeRoot: dimension mismatch across books.";
            }
            return false;
        }
        out->h_vec.push_back(book.cols);
    }

    out->offsets.assign(m, 0);
    int off = 0;
    for (int l = 0; l < m; ++l) {
        out->offsets[l] = off;
        off += out->h_vec[l];
    }

    const int h0 = out->h_vec[0];
    if (h0 <= 0) {
        if (err) {
            *err = "BuildPrecompLargeRoot: invalid h0.";
        }
        return false;
    }

    // C0 and norm0
    out->C0 = ColMajorMatrix<float>(d, h0);
    out->norm0.assign(static_cast<std::size_t>(h0), 0.0f);
    for (int c = 0; c < h0; ++c) {
        const float* src = C_root.books[0].Col(c);
        float* dst = out->C0.Col(c);
        std::copy(src, src + d, dst);
        out->norm0[static_cast<std::size_t>(c)] = Norm2(src, d);
    }

    // Flatten layers 1.. into C_small
    out->small_offsets.assign(m, 0);
    int Hs = 0;
    for (int l = 1; l < m; ++l) {
        out->small_offsets[l] = Hs;
        Hs += out->h_vec[l];
    }
    out->H_small = Hs;
    out->C_small = ColMajorMatrix<float>(d, Hs);
    out->invnorm_small_flat.assign(static_cast<std::size_t>(Hs), 0.0f);
    int col = 0;
    for (int l = 1; l < m; ++l) {
        const auto& book = C_root.books[l];
        for (int c = 0; c < book.cols; ++c) {
            const float* src = book.Col(c);
            float* dst = out->C_small.Col(col + c);
            std::copy(src, src + d, dst);
            const float n2 = Norm2(src, d);
            out->invnorm_small_flat[static_cast<std::size_t>(col + c)] =
                n2 > 0.0f ? 1.0f / std::sqrt(n2) : 0.0f;
        }
        col += book.cols;
    }

    out->G_small = ColMajorMatrix<float>(Hs, Hs);
    out->G0S = ColMajorMatrix<float>(h0, Hs);
    out->G0S_T = ColMajorMatrix<float>();
    {
        kernels->Gemm(true, false, 1.0f, out->C_small, out->C_small, 0.0f, &out->G_small);
        kernels->Gemm(true, false, 1.0f, out->C0, out->C_small, 0.0f, &out->G0S);
    }

    if (opts.build_g0s_transpose && h0 > 0 && Hs > 0) {
        const std::uint64_t bytes =
            static_cast<std::uint64_t>(sizeof(float)) *
            static_cast<std::uint64_t>(h0) *
            static_cast<std::uint64_t>(Hs);
        const std::uint64_t max_bytes =
            opts.g0s_transpose_max_mb > 0
                ? static_cast<std::uint64_t>(opts.g0s_transpose_max_mb) * 1024ULL * 1024ULL
                : 0ULL;
        if (max_bytes > 0 && bytes > max_bytes) {
            // Skip (leave G0S_T empty).
        } else {
            out->G0S_T = ColMajorMatrix<float>(Hs, h0);
            // Transpose: G0S is (h0×Hs) col-major, produce (Hs×h0) col-major.
            // For each small_flat column `flat`, read its contiguous h0 values and scatter into G0S_T columns.
            #pragma omp parallel for default(none) schedule(static) shared(out) firstprivate(Hs, h0)
            for (int flat = 0; flat < Hs; ++flat) {
                const float* src = out->G0S.Col(flat);  // length h0
                float* dst = out->G0S_T.data.data() + static_cast<std::size_t>(flat);
                for (int cid = 0; cid < h0; ++cid) {
                    dst[static_cast<std::size_t>(cid) * static_cast<std::size_t>(Hs)] = src[cid];
                }
            }
        }
    }

    // Bump a monotonic tag so cached backends can detect that the *contents* changed, even if
    // std::vector storage is reused across iterations.
    static std::uint64_t g_build_tag = 0;
    out->build_tag = ++g_build_tag;

    out->ready = true;
    return true;
}

void ComputeXcSmall(const PrecompLargeRoot& pre,
                    const ColMajorMatrix<float>& Xblk,
                    StreamKernelProvider* kernels,
                    ColMajorMatrix<float>* xC_small) {
    CpuStreamKernels cpu_kernels;
    if (!kernels) {
        kernels = &cpu_kernels;
    }
    xC_small->rows = pre.H_small;
    xC_small->cols = Xblk.cols;
    xC_small->data.assign(static_cast<std::size_t>(pre.H_small) *
                              static_cast<std::size_t>(Xblk.cols),
                          0.0f);
    kernels->Gemm(true, false, 1.0f, pre.C_small, Xblk, 0.0f, xC_small);
}

}  // namespace stlq
