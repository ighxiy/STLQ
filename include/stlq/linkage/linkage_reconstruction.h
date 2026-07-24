#pragma once

#include <cstdint>
#include <vector>

#include "stlq/common/types.h"

namespace stlq {

// Julia reference: compute_linkaged_reconstruction_multibook (demos/utils.jl:238-289).
//
// Semantics:
// - depth==0 nodes reconstruct from C_root using (B,a).
// - depth>0 nodes reconstruct a residual from C_one using (B,a), then add parent's reconstruction.
// - parent is 0 for root; otherwise 1-based global index (Julia-style).
ColMajorMatrix<float> ComputeLinkagedReconstructionMultiBook(const CodebookPack& C_root,
                                                            const CodebookPack& C_one,
                                                            const ColMajorMatrix<FullCode>& B,
                                                            const ColMajorMatrix<float>& a,
                                                            const std::vector<int>& parent,
                                                            const std::vector<int>& depth);

// Reconstruct using per-cluster linkage structure (preferred for IVF-style pipelines).
// Semantics match the parent/depth overload, but use LinkageStructure::Cluster local parent/depth ordering.
ColMajorMatrix<float> ComputeLinkagedReconstructionMultiBook(const CodebookPack& C_root,
                                                            const CodebookPack& C_one,
                                                            const ColMajorMatrix<FullCode>& B,
                                                            const ColMajorMatrix<float>& a,
                                                            const LinkageStructure& linkage);

// In-place version: overwrites `out` only for columns present in `linkage` (global indices).
// Typical use:
//   out = ReconstructAll(C_root.books, B, a);          // base reconstruction
//   ApplyLinkagedReconstructionMultiBookInPlace(...);   // overwrite linkaged nodes
void ApplyLinkagedReconstructionMultiBookInPlace(const CodebookPack& C_root,
                                                const CodebookPack& C_one,
                                                const ColMajorMatrix<FullCode>& B,
                                                const ColMajorMatrix<float>& a,
                                                const LinkageStructure& linkage,
                                                ColMajorMatrix<float>* out);

// Pointer-column variant:
// - `B_cols[i]` / `a_cols[i]` point to `m` entries for node `i` (0..n-1).
// - `out_cols[i]` points to a writable buffer of length `d` for node `i`.
// This lets callers write real nodes directly into a global output matrix while keeping
// virtual nodes in a temporary buffer, avoiding extra `d×n_aug` allocations.
void ApplyLinkagedReconstructionMultiBookInPlacePtrs(const CodebookPack& C_root,
                                                    const CodebookPack& C_one,
                                                    const FullCode* const* B_cols,
                                                    const float* const* a_cols,
                                                    int n,
                                                    const LinkageStructure& linkage,
                                                    float* const* out_cols);

// Pointer-column variant for non-contiguous storage (e.g., per-cluster real columns +
// separate virtual-node columns). `B_cols[i]` / `a_cols[i]` must point to `m` entries,
// where `m = C_root.books.size()`, and `linkage` indices must be within [0, n).
ColMajorMatrix<float> ComputeLinkagedReconstructionMultiBook(const CodebookPack& C_root,
                                                            const CodebookPack& C_one,
                                                            const FullCode* const* B_cols,
                                                            const float* const* a_cols,
                                                            int n,
                                                            const LinkageStructure& linkage);

}  // namespace stlq
