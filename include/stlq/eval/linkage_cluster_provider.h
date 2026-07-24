#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "stlq/eval/norm2_lut.h"
#include "stlq/io/linkage_coeff_codec_store.h"
#include "stlq/io/linkage_list_store.h"

namespace stlq::eval {

struct ClusterView {
    int cid = -1;
    int m = 0;
    int m_codes = 0;
    int n_real = 0;
    int n_virt = 0;
    int nc = 0;
    int n_root_real = 0;
    int depth_offsets_len = 0;  // number of entries in depth_offsets (>=2 when present)

    const std::uint32_t* real_ids = nullptr;      // n_real
    const std::uint32_t* parent_1based = nullptr; // n_real, non-null only for u32-backed clusters
    const std::uint16_t* parent_1based_u16 = nullptr; // n_real, non-null only for u16-backed clusters
    bool parent_is_u16 = false;
    const std::uint32_t* depth_offsets = nullptr; // >=2 (depth_offsets[1]==n_root_real)

    // Eval contract: codes are stored as uint8 payloads.
    // - codes_small_bytes: layers 1..m-1, length n_real*m_codes bytes
    // - code0_one_bytes:   length n_real bytes
    // - virt_codes_small_bytes: length n_virt*m_codes bytes
    const std::uint8_t* codes_small_bytes = nullptr;      // n_real*m_codes
    const std::uint8_t* code0_one_bytes = nullptr;        // n_real
    const std::uint8_t* virt_codes_small_bytes = nullptr; // n_virt*m_codes

    // Int8 coeff codec (optional, layer-major, length m*nc).
    const std::int8_t* q_layer_major = nullptr;
    const float* scales_root = nullptr;  // m
    const float* scales_linkage = nullptr; // m

    // Float coeffs (optional).
    const float* a0 = nullptr;            // n_real
    const float* coeffs_small = nullptr;  // n_real*m_codes
    const float* virt_a0 = nullptr;       // n_virt
    const float* virt_coeffs_small = nullptr; // n_virt*m_codes

    const float* r_norm2 = nullptr; // n_real
    const std::uint8_t* r_norm2_lut_u8 = nullptr;   // n_real
    const float* r_norm2_lut_centers = nullptr;     // r_norm2_lut_size
    int r_norm2_lut_size = 0;
};

// Interface for per-cluster norm2 computation. Implemented by the redesign’s NormProvider.
class IClusterNormProvider {
public:
    virtual ~IClusterNormProvider() = default;
    virtual bool ComputeNorm2(const ClusterView& cv,
                              std::vector<float>* out_r_norm2,
                              std::string* err) = 0;
    virtual bool ComputeNorm2LutFromHost(const float* norm2,
                                         int n,
                                         int requested_centers,
                                         int max_iter,
                                         Norm2Lut* out,
                                         std::string* err) {
        return BuildNorm2Lut(norm2, n, requested_centers, max_iter, out, err);
    }
    virtual bool IsThreadSafeForParallelPrecompute() const { return true; }
    virtual std::unique_ptr<IClusterNormProvider> CloneForParallelPrecompute() const { return nullptr; }
};

class ClusterProvider {
public:
    ClusterProvider() = default;

    struct PrepStats {
        double coeff_decode_sec = 0.0;  // Huffman CPU-decode-only (DecodeGroupIntoRanges, no I/O)
        double coeff_io_sec = 0.0;      // cthr.ReadCluster disk-read wall time (cache misses only)
        double norm_prep_sec = 0.0;     // norm2 compute time (cache misses only)
        double parent_louds_decode_sec = 0.0;  // LOUDS CPU-only decode: deser+decode+slice (no I/O)
        double parent_louds_read_sec = 0.0;    // ReadClusterParentLOUDSBlob disk-read wall (I/O only)
        double parent_louds_deser_sec = 0.0;   // ParentLOUDS::Deserialize only
        double parent_louds_decode_parent_sec = 0.0;  // ParentLOUDS::DecodeParent1Based only
        double parent_louds_slice_sec = 0.0;   // slice [n_virt, n_virt+n_real) into per-real parent
    };

    bool Open(const io::LinkageListReader& linkage_list,
              const io::LinkageCoeffCodecReader* coeff_codec,
              IClusterNormProvider* norm_provider,
              bool use_coeff_codec,
              bool prefer_parent_louds,
              std::uint32_t parent_louds_select_stride,
              std::uint32_t parent_louds_rank_words_per_super_log2,
              bool parent_louds_build_indices,
              bool adaptive_parent_u16_storage,
              bool use_norm2_lut,
              bool use_norm2_lut_global,
              int norm2_lut_h,
              int norm2_lut_kmeans_niter,
              bool profile_prep_stats,
              std::string* err);

    // Returns an immutable view backed by internal caches.
    // The view remains valid until this provider is destroyed.
    bool GetCluster(int cid, ClusterView* out, PrepStats* stats, std::string* err);

    // Timing statistics returned by PreloadAllClusters.
    // Contains ONLY pure CPU decode time (no I/O):
    //   louds_cpu_decode_sec : LOUDS Deserialize + DecodeParent1Based + parent-slice
    //   coeff_cpu_decode_sec : Huffman DecodeGroupIntoRanges only
    struct PreloadStats {
        double louds_cpu_decode_sec = 0.0;
        double coeff_cpu_decode_sec = 0.0;
    };

    // Benchmark result for BenchmarkDecodeWall().
    // Wall times are averages over bench_times iterations (OMP-parallel decode).
    struct BenchmarkResult {
        double louds_wall_avg_sec = 0.0;
        double coeff_wall_avg_sec = 0.0;
        // Profiling breakdown (filled only when request_profile_breakdown is true).
        // These are totals summed across all clusters for one profiling iteration.
        double profile_louds_read_sec = 0.0;
        double profile_louds_deser_sec = 0.0;
        double profile_louds_decode_parent_sec = 0.0;
        double profile_louds_slice_sec = 0.0;
    };

    // Pre-seed the norm2 cache from a precomputed map (loaded from disk).
    // Entries already in norm_cache_ are NOT overwritten.
    void PreloadNorm2Cache(std::unordered_map<int, std::vector<float>>&& preloaded);
    void PreloadNorm2LutCache(std::unordered_map<int, Norm2Lut>&& preloaded);
    void PreloadGlobalNorm2LutCache(Norm2Lut&& preloaded);
    void SetGlobalNorm2LutCenters(std::vector<float>&& centers);
    void SetNorm2DiskLazyEnabled(bool enabled);

    // Export the current norm2 cache (keyed by cid → r_norm2 vector) for storage.
    std::unordered_map<int, std::vector<float>> ExportNorm2Cache() const;
    std::unordered_map<int, Norm2Lut> ExportNorm2LutCache() const;

    // Compute r_norm2 for every non-empty cluster using streaming readers and return the full map.
    // This does not require the caller to touch clusters through GetCluster first.
    bool PrecomputeAllNorm2(std::unordered_map<int, std::vector<float>>* out,
                            int n_io_threads = 0,
                            std::string* err = nullptr);

    // Returns true if PreloadAllClusters has been successfully completed.
    bool IsPreloaded() const { return preloaded_; }

    // Preload ALL cluster list entries (and, if use_coeff_codec, all coeff entries) into the
    // in-memory caches before query evaluation begins.  This converts the per-query cold-start
    // I/O into a single sequential-order bulk load, which is dramatically faster for large nlist
    // (e.g. SIFT1B at nlist=65536).
    //
    // n_io_threads :
    //   < 0  → disabled (no-op, returns true immediately)
    //   = 0  → follow the process default OMP thread count (capped at nlist)
    //   > 0  → use exactly that many threads (capped at nlist)
    //
    // stats (optional) : receives CPU-only decode times (no I/O overhead) for reporting.
    //
    // Each thread owns its own LinkageListThreadReader / LinkageCoeffCodecThreadReader so that
    // file-handle seeking is fully independent and does not contend with the main linkage_thr_.
    bool PreloadAllClusters(int n_io_threads = 0, PreloadStats* stats = nullptr, std::string* err = nullptr);

    // Benchmark LOUDS + Huffman decode wall time on a SUBSET of clusters (the ones in cids).
    // LOUDS: re-reads raw blobs from disk (untimed pre-read), then bench_times OMP-parallel
    //        Deserialize+DecodeParent passes; reports average wall time.
    // Huffman: re-uses lens/payload already in coeff_cache_ (no disk I/O); bench_times
    //          OMP-parallel DecodeGroupIntoRanges passes; reports average wall time.
    // Requires that the clusters are already in list_cache_ / coeff_cache_.
    //
    // request_profile_breakdown : if true, run a single extra LOUDS profiling pass and fill
    //                             the per-field breakdown in out (deser_sec, decode_parent_sec,
    //                             slice_sec).  Slightly more expensive; opt-in.
    bool BenchmarkDecodeWall(const std::vector<int>& cids,
                             int bench_times,
                             bool request_profile_breakdown,
                             BenchmarkResult* out,
                             std::string* err);

    // Free raw Huffman payloads (lens_root/linkage + payload_root/linkage) from all cached
    // CoeffEntry objects.  The decoded q_layer_major and scales_root/linkage are kept because
    // the scan kernel still needs them.  Safe to call multiple times (clearing an already-empty
    // vector is a no-op).  Should be called after BenchmarkDecodeWall completes (delayed
    // release: bench needs the payloads) or immediately after preload when bench_times=0.
    // LOUDS raw blobs are never stored in the cache (freed inside LoadLinkageListEntryWith).
    void ReleaseCoeffRawPayloads();

    struct ParentStorageStats {
        std::uint64_t baseline_parent_u32_bytes = 0;   // hypothetical if all cached as uint32
        std::uint64_t resident_parent_bytes = 0;       // actual resident parent cache bytes
        int clusters_u16 = 0;
        int clusters_u32 = 0;
    };

    ParentStorageStats GetParentStorageStats() const;

private:
    struct LinkageListEntry {
        int cid = -1;
        int n_real = 0;
        int n_virt = 0;
        int nc = 0;
        int n_root_real = 0;
        double parent_louds_decode_sec = 0.0;
        double parent_louds_read_sec = 0.0;
        double parent_louds_deser_sec = 0.0;
        double parent_louds_decode_parent_sec = 0.0;
        double parent_louds_slice_sec = 0.0;
        std::vector<std::uint32_t> real_ids;
        std::vector<std::uint32_t> parent;
        std::vector<std::uint16_t> parent_u16;
        bool parent_stored_as_u16 = false;
        std::vector<std::uint32_t> depth_offsets;
        std::vector<std::uint8_t> codes_small;
        std::vector<std::uint8_t> code0_one;
        std::vector<std::uint8_t> virt_codes_small;

        // Float coeffs (optional).
        std::vector<float> a0;
        std::vector<float> coeffs_small_f32;
        std::vector<float> virt_a0;
        std::vector<float> virt_coeffs_small_f32;
    };

    struct CoeffEntry {
        int cid = -1;
        int nc = 0;
        std::vector<float> scales_root;
        std::vector<float> scales_linkage;
        std::vector<std::vector<std::uint8_t>> lens_root;
        std::vector<std::vector<std::uint8_t>> lens_linkage;
        std::vector<std::vector<std::uint8_t>> payload_root;
        std::vector<std::vector<std::uint8_t>> payload_linkage;
        std::vector<std::int8_t> q_layer_major;
        // CPU-only Huffman decode time (DecodeGroupIntoRanges only, not I/O).
        double cpu_decode_sec = 0.0;
        // Disk read wall time for cthr.ReadCluster (I/O only, cache misses only).
        double io_sec = 0.0;
    };

    struct NormEntry {
        int cid = -1;
        std::vector<float> r_norm2;
        Norm2Lut lut;
    };

    bool LoadLinkageListEntry(int cid, LinkageListEntry* out, std::string* err);
    bool LoadLinkageListEntryWith(io::LinkageListThreadReader& thr,
                                int cid, LinkageListEntry* out,
                                bool profile, std::string* err);
    void MaybeCompactParent(LinkageListEntry* out);

    bool LoadAndDecodeCoeff(int cid, const LinkageListEntry& list, CoeffEntry* out, std::string* err);
    bool LoadAndDecodeCoeffWith(io::LinkageCoeffCodecThreadReader& cthr,
                                int cid, const LinkageListEntry& list,
                                CoeffEntry* out, std::string* err);

    const io::LinkageListReader* linkage_list_ = nullptr;
    io::LinkageListThreadReader linkage_thr_;

    const io::LinkageCoeffCodecReader* coeff_reader_ = nullptr;
    io::LinkageCoeffCodecThreadReader coeff_thr_;

    IClusterNormProvider* norm_provider_ = nullptr;
    bool use_coeff_codec_ = false;
    bool prefer_parent_louds_ = true;
    std::uint32_t parent_louds_select_stride_ = 128;
    std::uint32_t parent_louds_rank_words_per_super_log2_ = 4;
    bool parent_louds_build_indices_ = true;
    bool adaptive_parent_u16_storage_ = false;
    bool use_norm2_lut_ = false;
    bool use_norm2_lut_global_ = false;
    int norm2_lut_h_ = 256;
    int norm2_lut_kmeans_niter_ = 25;
    bool norm2_disk_lazy_enabled_ = false;
    bool profile_prep_stats_ = false;
    std::vector<float> global_norm2_lut_centers_;

    std::unordered_map<int, LinkageListEntry> list_cache_;
    std::unordered_map<int, CoeffEntry> coeff_cache_;
    std::unordered_map<int, NormEntry> norm_cache_;
    bool preloaded_ = false;
};

}  // namespace stlq::eval
