#pragma once

#include <string>

#include "stlq/linkage/linkage_summary.h"
#include "stlq/common/config.h"
#include "stlq/common/types.h"

namespace stlq {

class StreamKernelProvider;

namespace io {
class BaseBasicReader;
class BaseListReader;
class IvfListsReader;
class LinkageListReader;
struct DatasetVectorReader;
}  // namespace io

// Update C_root from the streaming train_basic store + raw train vectors.
// Uses a two-pass scheme:
//  - pass1: solve small layers (l>=1) via flattened Gram on x_res = x_rot - a0*c0
//  - pass2: update root layer (l=0) via closed form on (x_rot - recon_small)
bool UpdateCRootFromTrainBasicStreaming(const Config& cfg,
                                       const std::string& train_basic_dir,
                                       const ColMajorMatrix<float>& R,
                                       const io::DatasetVectorReader* train_reader,
                                       StreamKernelProvider* kernels,
                                       CodebookPack* C_root_inout,
                                       float* out_mse_after_update,  // optional (computed using stored codes/coeffs)
                                       std::string* err);

// Update C_root from the streaming train_basic store + raw train vectors.
// Strictly equivalent to the full in-memory UpdateCodebooksLS solve on (l=0..m-1),
// by eliminating the root layer via a Schur complement per cluster (list).
// This preserves the exact least-squares optimum (up to floating-point rounding and the same diagonal
// bumping policy as the full solve), while avoiding construction of the huge (h0+Hs)×(h0+Hs) Gram matrix.
bool UpdateCRootFromTrainBasicStreamingExactLS(const Config& cfg,
                                              const std::string& train_basic_dir,
                                              const ColMajorMatrix<float>& R,
                                              const io::DatasetVectorReader* train_reader,
                                              StreamKernelProvider* kernels,
                                              CodebookPack* C_root_inout,
                                              float* out_mse_after_update,  // optional (computed using stored codes/coeffs)
                                              double* out_mse_wall_sec,      // optional (wall time spent in MSE scan)
                                              std::string* err);

// Update C_one from the disk train_linkage_list (depth-order) and train_list/raw (list-order).
// Only depth>0 real nodes participate.
bool UpdateCOneFromTrainLinkageListStreaming(const Config& cfg,
                                          const TrainResult& train,
                                          const io::BaseListReader& train_list,
                                          const io::IvfListsReader& ivf,
                                          const io::LinkageListReader& linkage_list,
                                          const io::DatasetVectorReader* fallback_reader,
                                          bool allow_random_fallback,
                                          StreamKernelProvider* kernels,
                                          CodebookPack* C_one_inout,
                                          std::string* err);

// Prompt 11 init-phase: update C_one from the init linkage_list built with only C_root.
// - The init linkage_list stores depth>0 residual encodings using C_root semantics (no C_one).
// - Layer0 codes are derived from a geometry-aware root_to_one mapping (SphericalKmeans on C_root[0]).
bool UpdateCOneFromInitLinkageListStreamingExactLS(const Config& cfg,
                                                 const TrainResult& train,  // uses train.C_root, train.R, train.is_bad_cluster
                                                 const io::BaseListReader& train_list,
                                                 const io::IvfListsReader& ivf,
                                                 const io::LinkageListReader& linkage_list,
                                                 const io::DatasetVectorReader* fallback_reader,
                                                 bool allow_random_fallback,
                                                 StreamKernelProvider* kernels,
                                                 CodebookPack* C_one_out,
                                                 std::string* err);

// Streaming OPQ update: accumulate cross-cov M = sum(Zhat * X^T) by cluster, then R = U*V^T.
bool UpdateOpqRotationStreamingByCluster(const Config& cfg,
                                        const TrainResult& train,
                                        const io::BaseListReader& train_list,
                                        const io::IvfListsReader& ivf,
                                        const io::LinkageListReader& linkage_list,
                                        const io::DatasetVectorReader* fallback_reader,
                                        bool allow_random_fallback,
                                        StreamKernelProvider* kernels,
                                        ColMajorMatrix<float>* R_inout,
                                        std::string* err);

// OPQ update from a pre-accumulated cross-covariance matrix M = sum(Zhat * X^T).
// This is useful to reuse an existing scan over train_list + linkage_list (e.g. during metrics analysis)
// and avoid re-reading/reconstructing when updating OPQ.
bool UpdateOpqRotationFromCrossCov(const ColMajorMatrix<float>& M,
                                  ColMajorMatrix<float>* R_inout,
                                  std::string* err);

// Optional training-stage analysis: compute linkage depth stats and reconstruction MSE using the on-disk linkage_list.
// This is intended for logging/debug; it performs additional scans and should be gated by config.train.log_metrics.
bool AnalyzeTrainLinkageListAfterCOneUpdateStreaming(const Config& cfg,
                                                  const TrainResult& train,  // uses train.C_root, train.C_one, train.R
                                                  const io::BaseListReader& train_list,
                                                  const io::IvfListsReader& ivf,
                                                  const io::LinkageListReader& linkage_list,
                                                  // When true, interpret `code0_one` bytes in the linkage_list as
                                                  // C_root layer-0 codes (root space) and map them to C_one[0] codes
                                                  // before reconstruction. This is used for the init-linkage list
                                                  // (single-codebook C_root-only build).
                                                  bool code0_one_bytes_are_root_codes,
                                                  const io::DatasetVectorReader* fallback_reader,
                                                  bool allow_random_fallback,
                                                  StreamKernelProvider* kernels,
                                                  LinkageDepthStats* depth_stats_out,
                                                  MseStats* mse_stats_out,
                                                  // Optional: when non-null, also accumulate the OPQ cross-covariance
                                                  // M = sum(Zhat * X^T) in a single pass. Zhat is the reconstructed
                                                  // rotated vectors (same as used for MSE), and X is the unrotated raw.
                                                  // Intended to avoid a second full scan in UpdateOpqRotationStreamingByCluster.
                                                  ColMajorMatrix<float>* opq_M_out,
                                                  std::string* err);

// Compute is_bad_cluster mask from the on-disk linkage_list by computing per-cluster mean MSE.
// Uses the same quantile-based threshold as BuildBadClusterMaskQuantile in the non-large pipeline.
// Should be called after UpdateCOneFromTrainLinkageListStreaming to use up-to-date codebooks.
bool ComputeBadClusterMaskFromLinkageListStreaming(const Config& cfg,
                                                  const TrainResult& train,  // uses train.C_root, train.C_one, train.R
                                                  const io::BaseListReader& train_list,
                                                  const io::IvfListsReader& ivf,
                                                  const io::LinkageListReader& linkage_list,
                                                  const io::DatasetVectorReader* fallback_reader,
                                                  bool allow_random_fallback,
                                                  StreamKernelProvider* kernels,
                                                  std::vector<bool>* is_bad_cluster_out,
                                                  double* baseline_out,  // optional
                                                  std::string* err);

// Prompt 11 init-phase: compute bad cluster mask from the init linkage_list built with only C_root.
// Uses C_root-only reconstruction for depth>0 nodes (no C_one).
bool ComputeBadClusterMaskFromInitLinkageListStreaming(const Config& cfg,
                                                     const TrainResult& train,  // uses train.C_root, train.R
                                                     const io::BaseListReader& train_list,
                                                     const io::IvfListsReader& ivf,
                                                     const io::LinkageListReader& linkage_list,
                                                     const io::DatasetVectorReader* fallback_reader,
                                                     bool allow_random_fallback,
                                                     StreamKernelProvider* kernels,
                                                     std::vector<bool>* is_bad_cluster_out,
                                                     double* baseline_out,  // optional
                                                     std::string* err);

}  // namespace stlq
