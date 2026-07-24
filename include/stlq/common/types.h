#pragma once

#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#if !defined(STLQ_ALWAYS_INLINE)
#if defined(_MSC_VER)
#define STLQ_ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define STLQ_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define STLQ_ALWAYS_INLINE inline
#endif
#endif

namespace stlq {

using Index = std::uint32_t;
// Small-layer code type (typically 8-bit when all h_vec[l>=1] <= 256).
using Code = std::uint8_t;
static_assert(sizeof(Code) == 1, "stlq: small-layer Code must be uint8_t (h_j<=256).");
// Root-layer code type: routing can exceed 65536 (e.g. IVF nlist>65536), so use uint32.
// NOTE: This does NOT change the on-disk store types. Only init-linkage legacy stores may write
// layer0 codes with an explicit `code0_width_bytes` (1/2/4).
using RootCode = std::uint32_t;
// Full code type for in-memory "m×n" code matrices (layers 1..m-1; non-root).
// Since all non-root layers satisfy h_j <= 256, FullCode = uint8_t suffices.
// Layer0 (root) codes with h0 > 256 must NOT be stored in FullCode; they go through a
// root sideband buffer (uint16_t when h0 <= 65536, uint32_t otherwise).
using FullCode = Code;  // == uint8_t
using Id = int;

template <typename T>
struct ColMajorMatrix {
    int rows = 0;
    int cols = 0;
    std::vector<T> data;

    ColMajorMatrix() = default;
    ColMajorMatrix(int r, int c) : rows(r), cols(c), data(static_cast<std::size_t>(r) * c) {}

    std::size_t Size() const { return data.size(); }

    T* Col(int j) { return data.data() + static_cast<std::size_t>(j) * rows; }
    const T* Col(int j) const { return data.data() + static_cast<std::size_t>(j) * rows; }

    T& operator()(int i, int j) { return data[static_cast<std::size_t>(j) * rows + i]; }
    const T& operator()(int i, int j) const { return data[static_cast<std::size_t>(j) * rows + i]; }
};

struct CodebookPack {
    int d = 0;
    std::vector<int> h_vec;
    std::vector<ColMajorMatrix<float>> books;
    // Monotonic build tag to let cached backends (e.g. CUDA) detect content changes even when
    // std::vector storage is reused across iterations.
    std::uint64_t build_tag = 0;
};

struct Precomp {
    ColMajorMatrix<float> C_all;
    ColMajorMatrix<float> G;
    std::vector<int> offsets;
    std::vector<int> flat_layer;
    std::vector<int> h_vec;
    std::vector<float> invnorm_flat;
    int H = 0;
    int d = 0;
    int m = 0;
    // Monotonic build tag to let cached backends (e.g. CUDA) detect content changes even when
    // std::vector storage is reused across iterations.
    std::uint64_t build_tag = 0;
};

struct BaseEncoding {
    ColMajorMatrix<FullCode> B;
    ColMajorMatrix<float> a;
};

// Virtual nodes encoding for virtual/base mode.
// Virtual nodes are stored separately from real base encodings to avoid (real+virt) concatenation.
// Global id convention (virtual mode only):
// - real ids:   [0, n_base_real)
// - virtual ids:[n_base_real, n_base_real + virt.B.cols), mapped by (g - n_base_real).
struct VirtualEncoding {
    int m = 0;
    int nlist = 0;
    // Per-cluster virtual encodings (cluster-local semantics).
    // Only bad clusters typically have virtual nodes; good clusters keep empty matrices.
    std::vector<ColMajorMatrix<FullCode>> B_by_cluster;   // length nlist
    std::vector<ColMajorMatrix<float>> a_by_cluster;  // length nlist

    int n_virtual(int cid) const {
        if (cid < 0 || cid >= nlist || B_by_cluster.empty()) {
            return 0;
        }
        return B_by_cluster[static_cast<std::size_t>(cid)].cols;
    }
    int n_virtual() const {
        int total = 0;
        for (const auto& Bc : B_by_cluster) total += Bc.cols;
        return total;
    }
};

// Ephemeral cache for bad-cluster in-cluster kNN (HNSW) tables used by:
// - virtual root augmentation (UMAP-like seed selection)
// - bad-cluster multi-center linkage building
//
// Tables store ONLY neighbor ids (no distances) in row-major [n_local x knn_k] using local ids.
// Lifetime: typically in-memory between `AddVirtualNodes(...)` and `BuildLinkageTwoCodebookVirtual(...)`.
struct BadClusterKnnCache {
    int nlist = 0;
    std::vector<int> knn_k_by_cluster;                 // length nlist (0 => missing)
    // Row-major [n_local * knn_k], storing cluster-local ids.
    std::vector<std::vector<std::uint32_t>> knn_ids_by_cluster;  // length nlist

    void Clear() {
        nlist = 0;
        knn_k_by_cluster.clear();
        knn_ids_by_cluster.clear();
    }
};

struct LinkageStructure {
    struct Cluster {
        int cluster_id = 0;

        // Cluster-local node layout (virtual-front):
        // - n_total = n_virtual + n_real
        // - local id range: [0, n_total)
        //   - virtual nodes: local in [0, n_virtual)
        //   - real nodes:    local in [n_virtual, n_total), real_idx = local - n_virtual
        int n_real = 0;                // number of real nodes in this cluster
        int n_virtual = 0;             // number of virtual nodes in this cluster

        // Real node global ids only (no virtual ids stored):
        // - indices.size() == n_real
        // - indices[real_idx] corresponds to local = n_virtual + real_idx
        // - global_id(local) = indices[local - n_virtual] (only when local is real)
        // - virtual nodes have no global id (not present in indices).
        std::vector<int> indices;

        // Per-node parent pointers in LOCAL space:
        // - parent_local.size() == n_total
        // - parent_local[local] is 1-based local id of the parent, or 0 for root (super-root).
        // - virtual nodes must be roots: parent_local[v] == 0 for v in [0, n_virtual).
        std::vector<int> parent_local;

        // Depth offsets for REAL nodes only (in real_idx space [0, n_real)):
        // - depth_offsets[0] == 0
        // - depth_offsets.back() == n_real
        // - depth_offsets[1] is the number of real roots (depth==0 among real nodes), if n_real>0.
        // The real nodes are stored in BFS / nondecreasing depth order (virtual nodes are separate prefix).
        std::vector<int> depth_offsets;
    };

    std::vector<Cluster> clusters;
    int max_depth = 0;
};

inline int LinkageClusterTotalNodes(const LinkageStructure::Cluster& c) {
    return c.n_real + c.n_virtual;
}
inline bool LinkageIsVirtualLocal(int local, int n_virt) {
    return local < n_virt;
}
inline int LinkageLocalToRealIndex(int local, int n_virt) {
    return local - n_virt;  // requires local>=n_virt
}
inline int LinkageRealIndexToLocal(int real_idx, int n_virt) {
    return n_virt + real_idx;
}

// Per-cluster cache used by virtual training for bad clusters (UMAP + multi-center linkageing).
// Real nodes are referenced by global ids in `cols_global` (size n_real).
// Virtual nodes are not part of global (state) encodings, so we keep their reconstruction only.
// `linkage_aug` is cluster-local with virtual-front local ids and indices for real nodes only.
struct BadClusterCache {
    int cluster_id = -1;
    int n_real = 0;
    int n_virt = 0;
    std::vector<int> cols_global;
    ColMajorMatrix<float> R_virt;  // d x n_virt, virtual roots reconstruction (depth==0)
    LinkageStructure linkage_aug;
};

struct Dataset {
    ColMajorMatrix<float> Xt;
    ColMajorMatrix<float> Xb;
    ColMajorMatrix<float> Xq;
    std::vector<int> gt;
};

struct TrainResult {
    CodebookPack C_root;
    CodebookPack C_one;
    ColMajorMatrix<float> R;
    std::vector<bool> is_bad_cluster;
    LinkageStructure train_linkage;

    // Optional training metadata for reproduction/debugging.
    // These are reported during training and saved into train-result metadata.
    // Note: `loaded_train_h5`/`saved_train_h5` are operational legacy field names.
    std::string loaded_train_h5;
    std::string saved_train_h5;
    // Optional hint for where large-pipeline intermediates (train_basic/train_ivf/train_list/...) were written.
    // This is stored in some checkpoints (e.g. pre-init-linkage) to support resuming without rebuilding basics.
    std::string exp_root_hint;
    float init_mse = -1.0f;
    float beam_mse = -1.0f;
    float icm_mse = -1.0f;
    float linkage_mse_mean = -1.0f;
    float linkage_ratio = -1.0f;
    int linkage_max_depth = -1;
    float linkage_mean_depth = -1.0f;
    int R_iters = 0;
    float final_linkaged_mse_rot = -1.0f;
    // Resume signature for checkpoint compatibility checks (0 => missing/unknown).
    std::uint64_t resume_sig_u64 = 0;
    // Checkpoint stage marker (0 => normal/full checkpoint; 1 => pre-init-linkage checkpoint that has C_root+R only).
    int checkpoint_stage = 0;
};

}  // namespace stlq
