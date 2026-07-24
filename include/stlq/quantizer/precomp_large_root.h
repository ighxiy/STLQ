#pragma once

#include <string>
#include <vector>

#include "stlq/common/types.h"

namespace stlq {

class StreamKernelProvider;

struct PrecompLargeRoot {
    int d = 0;
    int m = 0;
    std::vector<int> h_vec;
    std::vector<int> offsets; // original flat offsets (layer0 is large)
    // Monotonic build tag to let cached backends (e.g. CUDA) detect content changes even when
    // std::vector storage is reused across iterations.
    std::uint64_t build_tag = 0;

    // layer0
    ColMajorMatrix<float> C0;       // d×h0
    std::vector<float> norm0;       // h0 (squared L2)

    // layers 1..m-1 flattened into C_small
    int H_small = 0;
    std::vector<int> small_offsets; // size m, small_offsets[0]=0
    ColMajorMatrix<float> C_small;  // d×H_small
    ColMajorMatrix<float> G_small;  // H_small×H_small
    std::vector<float> invnorm_small_flat; // H_small

    // cross: layer0 vs small
    ColMajorMatrix<float> G0S;      // h0×H_small (C0' * C_small)
    // Optional CPU cache: transpose of G0S for cache-friendly access when h0 is large.
    // Layout: H_small×h0 (column-major). Column `cid` is contiguous over small_flat indices.
    ColMajorMatrix<float> G0S_T;

    bool ready = false;
};

struct PrecompLargeRootBuildOptions {
    // Build G0S_T (transpose of G0S). This can speed up CPU ICM/ILS when h0 is large
    // by turning strided reads into contiguous reads, at the cost of extra host RAM
    // roughly equal to sizeof(float)*h0*H_small.
    bool build_g0s_transpose = false;
    // Optional cap (MiB) for the transpose allocation. 0 disables the cap.
    int g0s_transpose_max_mb = 0;
};

// Build precomp for "large root" where h_vec[0] is big (e.g. 8192/16384).
// Keeps semantics but avoids allocating full xC(H×n).
bool BuildPrecompLargeRoot(const CodebookPack& C_root,
                           StreamKernelProvider* kernels,
                           const PrecompLargeRootBuildOptions& opts,
                           PrecompLargeRoot* out,
                           std::string* err);

// xC_small = C_small' * Xblk  (H_small×nblk)
void ComputeXcSmall(const PrecompLargeRoot& pre,
                    const ColMajorMatrix<float>& Xblk,
                    StreamKernelProvider* kernels,
                    ColMajorMatrix<float>* xC_small);

}  // namespace stlq
