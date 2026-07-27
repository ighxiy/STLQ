#include "stlq/eval/recall_linkage_disk.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "stlq/pipeline/large_store_hash.h"
#include "stlq/coeff/bit_io.h"
#include "stlq/coeff/huffman_canonical.h"
#include "stlq/core/blas.h"
#include "stlq/core/model_limits.h"
#include "stlq/core/threading.h"
#include "stlq/eval/linkage_norm_provider.h"
#include "stlq/eval/linkage_norm_provider_cuda.h"
#include "stlq/eval/norm2_lut.h"
#include "stlq/eval/query_table_builder.h"
#include "stlq/io/linkage_coeff_codec_store.h"
#include "stlq/io/linkage_list_store.h"
#include "stlq/ivf/cluster_select.h"
#include "stlq/ivf/ivf_probe_hnsw_index.h"
#include "stlq/common/logger.h"
#include "stlq/quantizer/codebook_meta.h"
#include "stlq/succinct/parent_louds.h"
#include "stlq/common/timer.h"

#include <omp.h>

namespace stlq {
namespace {

std::filesystem::path Norm2CacheHashPath(const std::filesystem::path& norm2_path) {
    return norm2_path.string() + ".hash.u64";
}

std::filesystem::path Norm2CacheCountPath(const std::filesystem::path& norm2_path) {
    return norm2_path.string() + ".count.u64";
}

static std::string TrimStr(std::string s) {
    std::size_t lo = 0;
    while (lo < s.size() && std::isspace(static_cast<unsigned char>(s[lo]))) ++lo;
    std::size_t hi = s.size();
    while (hi > lo && std::isspace(static_cast<unsigned char>(s[hi - 1]))) --hi;
    return s.substr(lo, hi - lo);
}

static bool StartsWithStr(const std::string& s, const char* p) {
    const std::size_t n = std::strlen(p);
    return s.size() >= n && s.compare(0, n, p) == 0;
}

static bool ParseKeyValueLine(const std::string& line_in, std::string* out_k, std::string* out_v) {
    if (!out_k || !out_v) return false;
    *out_k = {};
    *out_v = {};
    std::string line = TrimStr(line_in);
    if (line.empty()) return false;
    if (StartsWithStr(line, "#") || StartsWithStr(line, "//")) return false;

    const std::size_t hash_pos = line.find('#');
    const std::size_t slash_pos = line.find("//");
    std::size_t cut = std::string::npos;
    if (hash_pos != std::string::npos) cut = hash_pos;
    if (slash_pos != std::string::npos) cut = (cut == std::string::npos) ? slash_pos : std::min(cut, slash_pos);
    if (cut != std::string::npos) {
        line = TrimStr(line.substr(0, cut));
        if (line.empty()) return false;
    }

    const std::size_t eq = line.find('=');
    if (eq == std::string::npos) return false;
    std::string k = TrimStr(line.substr(0, eq));
    std::string v = TrimStr(line.substr(eq + 1));
    if (k.empty()) return false;
    if (v.size() >= 2) {
        const char q0 = v.front();
        const char q1 = v.back();
        if ((q0 == '"' && q1 == '"') || (q0 == '\'' && q1 == '\'')) {
            v = v.substr(1, v.size() - 2);
        }
    }
    *out_k = std::move(k);
    *out_v = std::move(v);
    return true;
}

static bool ReadMetaFile(const std::filesystem::path& path,
                         std::unordered_map<std::string, std::string>* out) {
    if (!out) return false;
    out->clear();
    std::ifstream in(path);
    if (!in) return false;
    std::string line;
    while (std::getline(in, line)) {
        std::string k, v;
        if (!ParseKeyValueLine(line, &k, &v)) continue;
        (*out)[k] = v;
    }
    return true;
}

static bool WriteMetaFile(const std::filesystem::path& path,
                          const std::vector<std::pair<std::string, std::string>>& kv) {
    std::ofstream out(path);
    if (!out) return false;
    for (const auto& [k, v] : kv) out << k << "=" << v << "\n";
    out.flush();
    return static_cast<bool>(out);
}

static std::uint64_t Mix64(std::uint64_t h, std::uint64_t v) {
    h ^= v;
    h ^= h >> 30;
    h *= 0xbf58476d1ce4e5b9ULL;
    h ^= h >> 27;
    h *= 0x94d049bb133111ebULL;
    h ^= h >> 31;
    return h;
}

static std::string ToHexU64(std::uint64_t v) {
    std::ostringstream oss;
    oss << "0x" << std::hex << v;
    return oss.str();
}

static bool ParseU64Auto(const std::string& s, std::uint64_t* out) {
    if (!out) return false;
    std::string t = TrimStr(s);
    if (t.empty()) return false;
    std::uint64_t v = 0;
    if (t.size() > 2 && t[0] == '0' && (t[1] == 'x' || t[1] == 'X')) {
        std::istringstream iss(t);
        iss >> std::hex >> v;
        if (!iss) return false;
    } else {
        try {
            v = static_cast<std::uint64_t>(std::stoull(t));
        } catch (...) {
            return false;
        }
    }
    *out = v;
    return true;
}

static bool ReadBinaryExact(const std::filesystem::path& path, void* out, std::size_t bytes) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    in.read(reinterpret_cast<char*>(out), static_cast<std::streamsize>(bytes));
    return static_cast<bool>(in);
}

static bool WriteBinaryExact(const std::filesystem::path& path, const void* data, std::size_t bytes) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    out.flush();
    return static_cast<bool>(out);
}

static std::filesystem::path DefaultHier2SplitCacheDirFromLinkageListDir(const std::filesystem::path& linkage_list_dir) {
    // linkage_list_dir is typically: <out_dir>/linkage_list
    return linkage_list_dir.parent_path() / "ivf_hier2_split";
}

static bool TryLoadHier2SplitCache(const std::filesystem::path& cache_dir,
                                  int d,
                                  int nlist,
                                  int K1,
                                  int kmeans_iters,
                                  unsigned seed,
                                  std::uint64_t centroids_hash,
                                  ivf::Hier2Split* out) {
    if (!out) return false;
    *out = {};
    const std::filesystem::path meta_path = cache_dir / "meta.txt";
    const std::filesystem::path assign_path = cache_dir / "assign.u32";
    std::error_code ec;
    if (!std::filesystem::exists(meta_path, ec) || ec) return true;
    if (!std::filesystem::exists(assign_path, ec) || ec) return true;

    const std::uint64_t expected_bytes = static_cast<std::uint64_t>(nlist) * sizeof(std::uint32_t);
    if (std::filesystem::file_size(assign_path, ec) != expected_bytes || ec) return true;

    std::unordered_map<std::string, std::string> meta;
    if (!ReadMetaFile(meta_path, &meta)) return true;
    auto get = [&](const char* k) -> std::string {
        auto it = meta.find(k);
        return (it == meta.end()) ? std::string() : it->second;
    };

    auto parse_i32 = [](const std::string& s, int* out_i) -> bool {
        if (!out_i) return false;
        try {
            *out_i = std::stoi(TrimStr(s));
            return true;
        } catch (...) {
            return false;
        }
    };

    int version = 0, md = 0, mK = 0, mK1 = 0, miters = 0, mseed = 0, mspherical = 0;
    std::uint64_t mhash = 0;
    if (!parse_i32(get("version"), &version) || version != 1) return true;
    if (!parse_i32(get("d"), &md) || md != d) return true;
    if (!parse_i32(get("K"), &mK) || mK != nlist) return true;
    if (!parse_i32(get("K1"), &mK1) || mK1 != K1) return true;
    if (!parse_i32(get("kmeans_iters"), &miters) || miters != kmeans_iters) return true;
    if (!parse_i32(get("seed"), &mseed) || static_cast<unsigned>(mseed) != seed) return true;
    if (!parse_i32(get("spherical"), &mspherical) || mspherical != 1) return true;
    if (!ParseU64Auto(get("centroids_hash"), &mhash) || mhash != centroids_hash) return true;

    std::vector<std::uint32_t> assign(static_cast<std::size_t>(nlist), 0);
    if (!ReadBinaryExact(assign_path, assign.data(), assign.size() * sizeof(std::uint32_t))) return true;

    ivf::Hier2Split s;
    s.K = nlist;
    s.K1 = K1;
    s.groups.resize(static_cast<std::size_t>(K1));
    for (int fid = 0; fid < nlist; ++fid) {
        const std::uint32_t c = assign[static_cast<std::size_t>(fid)];
        if (c < static_cast<std::uint32_t>(K1)) {
            s.groups[static_cast<std::size_t>(c)].push_back(fid);
        }
    }
    *out = std::move(s);
    return true;
}

static void TryWriteHier2SplitCache(const std::filesystem::path& cache_dir,
                                   int d,
                                   int nlist,
                                   const ivf::Hier2Split& split,
                                   int kmeans_iters,
                                   unsigned seed,
                                   std::uint64_t centroids_hash) {
    std::error_code ec;
    std::filesystem::create_directories(cache_dir, ec);
    if (ec) return;
    std::vector<std::uint32_t> assign(static_cast<std::size_t>(nlist), 0);
    for (int c = 0; c < split.K1; ++c) {
        for (int fid : split.groups[static_cast<std::size_t>(c)]) {
            if (fid >= 0 && fid < nlist) assign[static_cast<std::size_t>(fid)] = static_cast<std::uint32_t>(c);
        }
    }
    (void)WriteBinaryExact(cache_dir / "assign.u32", assign.data(), assign.size() * sizeof(std::uint32_t));
    (void)WriteMetaFile(cache_dir / "meta.txt",
                        {
                            {"version", "1"},
                            {"d", std::to_string(d)},
                            {"K", std::to_string(nlist)},
                            {"K1", std::to_string(split.K1)},
                            {"kmeans_iters", std::to_string(std::max(1, kmeans_iters))},
                            {"seed", std::to_string(static_cast<int>(seed))},
                            {"spherical", "1"},
                            {"centroids_hash", ToHexU64(centroids_hash)},
                        });
}

std::uint64_t CountNonEmptyRealClusters(const io::LinkageListReader& linkage_list) {
    const auto& real_offs = linkage_list.real_offsets();
    const int nlist = linkage_list.nlist();
    if (real_offs.size() != static_cast<std::size_t>(nlist + 1)) return 0;
    std::uint64_t count = 0;
    for (int cid = 0; cid < nlist; ++cid) {
        if (real_offs[static_cast<std::size_t>(cid + 1)] >
            real_offs[static_cast<std::size_t>(cid)]) {
            ++count;
        }
    }
    return count;
}

bool ReadSourceHashForNorm2Cache(const io::LinkageListReader& linkage_list,
                                 bool use_coeff_codec,
                                 std::uint64_t* out_hash) {
    if (!out_hash) return false;
    return app::ReadNorm2SourceHash(linkage_list.dir(), use_coeff_codec, out_hash);
}

struct Candidate {
    float dist = 0.0f;
    std::int32_t id = -1;
};

STLQ_ALWAYS_INLINE bool CandLess(const Candidate& a, const Candidate& b) {
    if (a.dist < b.dist) return true;
    if (a.dist > b.dist) return false;
    return a.id < b.id;
}

struct TopKHeap {
    int k = 0;
    std::vector<Candidate> heap;
    bool heapified = false;

    explicit TopKHeap(int k_in) : k(std::max(1, k_in)) {
        heap.reserve(static_cast<std::size_t>(k));
    }

    STLQ_ALWAYS_INLINE void SiftDown(int i) {
        const int n = static_cast<int>(heap.size());
        Candidate x = heap[static_cast<std::size_t>(i)];
        while (true) {
            const int l = i * 2 + 1;
            if (l >= n) break;
            int c = l;
            const int r = l + 1;
            if (r < n && CandLess(heap[static_cast<std::size_t>(c)], heap[static_cast<std::size_t>(r)])) {
                c = r;
            }
            if (!CandLess(x, heap[static_cast<std::size_t>(c)])) break;
            heap[static_cast<std::size_t>(i)] = heap[static_cast<std::size_t>(c)];
            i = c;
        }
        heap[static_cast<std::size_t>(i)] = x;
    }

    STLQ_ALWAYS_INLINE void Heapify() {
        for (int i = (static_cast<int>(heap.size()) >> 1) - 1; i >= 0; --i) {
            SiftDown(i);
        }
        heapified = true;
    }

    STLQ_ALWAYS_INLINE void Push(float dist, std::int32_t id) {
        const Candidate x{dist, id};
        if (static_cast<int>(heap.size()) < k) {
            heap.push_back(x);
            if (static_cast<int>(heap.size()) == k) Heapify();
            return;
        }
        if (!heapified) Heapify();
        const Candidate& worst = heap[0];
        if (x.dist > worst.dist || (x.dist == worst.dist && x.id >= worst.id)) return;
        heap[0] = x;
        SiftDown(0);
    }

    STLQ_ALWAYS_INLINE void PushAssumeBetter(float dist, std::int32_t id) {
        heap[0] = Candidate{dist, id};
        SiftDown(0);
    }

    void Finalize(float* out_dists, int* out_ids) {
        std::sort(heap.begin(), heap.end(), [](const Candidate& a, const Candidate& b) {
            return CandLess(a, b);
        });
        const int got = static_cast<int>(heap.size());
        for (int i = 0; i < got; ++i) {
            out_dists[i] = heap[static_cast<std::size_t>(i)].dist;
            out_ids[i] = static_cast<int>(heap[static_cast<std::size_t>(i)].id);
        }
        for (int i = got; i < k; ++i) {
            out_dists[i] = std::numeric_limits<float>::infinity();
            out_ids[i] = -1;
        }
    }
};

// CodeBytes is always 1: one/root-small layers use uint8 codes (guaranteed by design).
STLQ_ALWAYS_INLINE std::uint8_t ReadSmallCode(const std::uint8_t* base, int pos, int m_codes, int idx) {
    const std::ptrdiff_t off =
        static_cast<std::ptrdiff_t>(pos) * static_cast<std::ptrdiff_t>(m_codes) +
        static_cast<std::ptrdiff_t>(idx);
    return base[off];
}

inline float Dot(const float* a, const float* b, int n) {
    float s = 0.0f;
    #pragma omp simd reduction(+:s)
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

inline float Norm2(const float* a, int n) {
    float s = 0.0f;
    #pragma omp simd reduction(+:s)
    for (int i = 0; i < n; ++i) s += a[i] * a[i];
    return s;
}

inline std::int8_t QFromSym(std::uint16_t sym, int base) {
    return std::int8_t(int(sym) - base);
}

void DecodeAllLayersIntoRanges(int m,
                               int nc,
                               std::int8_t* q_layer_major_out,
                               int base,
                               const HuffmanCanonicalModel& model,
                               const std::vector<std::uint8_t>& payload,
                               int r0_begin,
                               int r0_end,
                               int r1_begin,
                               int r1_end) {
    BitReader br(payload.data(), payload.size());
    for (int l = 0; l < m; ++l) {
        std::int8_t* ql = q_layer_major_out + static_cast<std::size_t>(l) * static_cast<std::size_t>(nc);
        for (int i = r0_begin; i < r0_end; ++i) {
            ql[i] = QFromSym(model.DecodeSymbol(br), base);
        }
        for (int i = r1_begin; i < r1_end; ++i) {
            ql[i] = QFromSym(model.DecodeSymbol(br), base);
        }
    }
}

void DecodeOneLayerIntoRanges(int nc,
                              std::int8_t* q_layer_out,
                              int base,
                              const HuffmanCanonicalModel& model,
                              const std::vector<std::uint8_t>& payload,
                              int r0_begin,
                              int r0_end,
                              int r1_begin,
                              int r1_end) {
    (void)nc;
    BitReader br(payload.data(), payload.size());
    for (int i = r0_begin; i < r0_end; ++i) q_layer_out[i] = QFromSym(model.DecodeSymbol(br), base);
    for (int i = r1_begin; i < r1_end; ++i) q_layer_out[i] = QFromSym(model.DecodeSymbol(br), base);
}

void DecodeGroupIntoRanges(int m,
                           int nc,
                           int streams,
                           int S,
                           int base,
                           const std::vector<std::vector<std::uint8_t>>& lens,
                           const std::vector<std::vector<std::uint8_t>>& payload,
                           int r0_begin,
                           int r0_end,
                           int r1_begin,
                           int r1_end,
                           std::int8_t* q_layer_major_out) {
    if (r0_begin >= r0_end && r1_begin >= r1_end) return;
    if (streams == 1) {
        HuffmanCanonicalModel model;
        model.S = static_cast<std::uint16_t>(S);
        model.len = lens[0];
        model.BuildDerivedTables();
        DecodeAllLayersIntoRanges(m, nc, q_layer_major_out, base, model, payload[0],
                                  r0_begin, r0_end, r1_begin, r1_end);
        return;
    }
    for (int l = 0; l < m; ++l) {
        HuffmanCanonicalModel model;
        model.S = static_cast<std::uint16_t>(S);
        model.len = lens[static_cast<std::size_t>(l)];
        model.BuildDerivedTables();
        std::int8_t* ql = q_layer_major_out + static_cast<std::size_t>(l) * static_cast<std::size_t>(nc);
        DecodeOneLayerIntoRanges(nc, ql, base, model, payload[static_cast<std::size_t>(l)],
                                 r0_begin, r0_end, r1_begin, r1_end);
    }
}

STLQ_ALWAYS_INLINE void CopyScalePackedTable(float* dst,
                                               const float* src,
                                               const int* offsets,
                                               int total_cols,
                                               const float* scales,
                                               int m,
                                               int start_layer) {
    const int prefix = (start_layer < m) ? offsets[start_layer] : total_cols;
    if (prefix > 0) {
        std::memcpy(dst, src, static_cast<std::size_t>(prefix) * sizeof(float));
    }
    for (int l = start_layer; l < m; ++l) {
        const int begin = offsets[l];
        const int end = (l + 1 < m) ? offsets[l + 1] : total_cols;
        const float s = scales[l];
        const float* ps = src + begin;
        float* pd = dst + begin;
        const int len = end - begin;
        #pragma omp simd
        for (int i = 0; i < len; ++i) pd[i] = s * ps[i];
    }
}

struct LoudsNativeCluster {
    int cid = -1;
    int m = 0;
    int m_codes = 0;
    int n_real = 0;
    int n_virt = 0;
    int nc = 0;
    int n_root_real = 0;

    std::vector<std::uint32_t> real_ids;
    std::vector<std::uint32_t> depth_offsets;
    std::vector<std::uint8_t> codes_small;
    std::vector<std::uint8_t> code0_one;
    std::vector<std::uint8_t> virt_codes_small;

    std::vector<float> a0;
    std::vector<float> coeffs_small;
    std::vector<float> virt_a0;
    std::vector<float> virt_coeffs_small;

    std::vector<float> scales_root;
    std::vector<float> scales_linkage;
    std::vector<std::vector<std::uint8_t>> lens_root;
    std::vector<std::vector<std::uint8_t>> lens_linkage;
    std::vector<std::vector<std::uint8_t>> payload_root;
    std::vector<std::vector<std::uint8_t>> payload_linkage;
    std::vector<std::int8_t> q_layer_major;

    std::vector<float> r_norm2;
    eval::Norm2Lut norm2_lut;
    std::vector<std::uint8_t> louds_blob;
    succinct::ParentLOUDS louds;
};

struct LoudsNativeBundle {
    const io::LinkageListReader* linkage_list = nullptr;
    io::LinkageListThreadReader linkage_thr;
    io::LinkageCoeffCodecReader coeff_reader;
    io::LinkageCoeffCodecThreadReader coeff_thr;
    std::unique_ptr<eval::IClusterNormProvider> norm_owned;
    bool use_coeff_codec = false;
    const ColMajorMatrix<float>* C_root0 = nullptr;
    CodebookMeta meta_root_small;
    CodebookMeta meta_one;
    ColMajorMatrix<float> D_rr;
    ColMajorMatrix<float> D_oo;
    ColMajorMatrix<float> D_ro;
    std::unordered_map<int, LoudsNativeCluster> cluster_cache;
    std::unordered_map<int, std::vector<float>> norm2_cache;
    std::unordered_map<int, eval::Norm2Lut> norm2_lut_cache;
    std::vector<float> global_norm2_lut_centers;
    bool norm2_disk_present = false;
    bool norm2_lut_disk_present = false;
    bool norm2_disk_lazy_enabled = false;
    bool norm2_lut_disk_lazy_enabled = false;
    bool norm2_lut_global = false;
    bool preload_done = false;
    bool setup_logged = false;
    bool load_mode_logged = false;
    bool hier2_logged = false;
    bool hnsw_logged = false;

    // Eval-only IVF probe HNSW index (over unit-normalized root centroids).
    std::shared_ptr<ivf::IvfProbeHnswIndex> ivf_probe_hnsw;
    bool timing_reported_louds = false;
    bool timing_reported_huffman = false;
    int preload_io_threads_used = 0;
    // True wall time for the whole preload phase (if enabled).
    double preload_wall_sec = 0.0;

    // Preload-mode component timing (critical-path wall; approximated by max per-thread sums).
    double preload_louds_wall_sec = 0.0;     // louds I/O + deser
    double preload_louds_cpu_sec = 0.0;      // louds deser only
    double preload_huffman_wall_sec = 0.0;   // coeff I/O + decode
    double preload_huffman_cpu_sec = 0.0;    // coeff decode only

    // Wall time spent in on-demand cache-miss loads.
    // In this LOUDS-native implementation, cache-miss loads are executed on the main thread (serial).
    double lazy_louds_wall_sec = 0.0;
    double lazy_huffman_wall_sec = 0.0;

    // CPU-only decode wall spent in on-demand cache-miss loads.
    double lazy_louds_cpu_sec = 0.0;
    double lazy_huffman_cpu_sec = 0.0;
};

static std::unique_ptr<eval::IClusterNormProvider> MakeNormProviderOwnedNative(
    const Config& cfg,
    const ColMajorMatrix<float>& C_root0,
    const CodebookMeta& meta_root_small,
    const CodebookMeta& meta_one) {
    auto norm_cpu_owned = std::make_unique<eval::LinkageNormProviderLookup>(
        C_root0, meta_root_small, meta_one);
#if !defined(STLQ_ENABLE_CUDA) || !STLQ_ENABLE_CUDA
    (void)cfg;
#endif
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
    if (cfg.eval.linkage_gpu_norm_enable && cfg.runtime.use_cuda) {
        return std::make_unique<eval::cuda::LinkageNormProviderLookupCuda>(
            C_root0,
            meta_root_small,
            meta_one,
            cfg.eval.linkage_gpu_scan_max_nc,
            /*profile_breakdown=*/false);
    }
#endif
    return norm_cpu_owned;
}

bool BuildProviderClusterView(bool use_coeff_codec,
                              const LoudsNativeBundle& bundle,
                              const LoudsNativeCluster& cl,
                              std::vector<std::uint32_t>* parent_1based,
                              eval::ClusterView* cv,
                              std::string* err) {
    if (!parent_1based || !cv) return false;
    parent_1based->assign(static_cast<std::size_t>(cl.n_real), 0u);
    if (cl.n_real > cl.n_root_real) {
        auto parent_decoder = cl.louds.MakeSequentialParentDecoder();
        parent_decoder.Skip(static_cast<std::size_t>(cl.n_virt + cl.n_root_real));
        for (int pos = cl.n_root_real; pos < cl.n_real; ++pos) {
            (*parent_1based)[static_cast<std::size_t>(pos)] =
                static_cast<std::uint32_t>(parent_decoder.NextParent1Based());
        }
    }

    *cv = eval::ClusterView{};
    cv->cid = cl.cid;
    cv->m = cl.m;
    cv->m_codes = cl.m_codes;
    cv->n_real = cl.n_real;
    cv->n_virt = cl.n_virt;
    cv->nc = cl.nc;
    cv->n_root_real = cl.n_root_real;
    cv->depth_offsets_len = static_cast<int>(cl.depth_offsets.size());
    cv->real_ids = cl.real_ids.data();
    cv->parent_is_u16 = false;
    cv->parent_1based = parent_1based->data();
    cv->parent_1based_u16 = nullptr;
    cv->depth_offsets = cl.depth_offsets.data();
    cv->codes_small_bytes = cl.codes_small.data();
    cv->code0_one_bytes = cl.code0_one.data();
    cv->virt_codes_small_bytes = cl.virt_codes_small.data();
    if (use_coeff_codec) {
        if (cl.q_layer_major.empty() || cl.scales_root.empty() || cl.scales_linkage.empty()) {
            if (err) *err = "LOUDS-native eval: missing quantized coeffs for LUT build.";
            return false;
        }
        cv->q_layer_major = cl.q_layer_major.data();
        cv->scales_root = cl.scales_root.data();
        cv->scales_linkage = cl.scales_linkage.data();
    } else {
        if (cl.a0.empty() || cl.coeffs_small.empty() ||
            (cl.n_virt > 0 && (cl.virt_a0.empty() || cl.virt_coeffs_small.empty()))) {
            if (err) *err = "LOUDS-native eval: missing float coeffs for LUT build.";
            return false;
        }
        cv->a0 = cl.a0.data();
        cv->coeffs_small = cl.coeffs_small.data();
        cv->virt_a0 = cl.virt_a0.data();
        cv->virt_coeffs_small = cl.virt_coeffs_small.data();
    }
    (void)bundle;
    return true;
}

bool ComputeNorm2LutNative(bool use_coeff_codec,
                           const Config& cfg,
                           const LoudsNativeBundle& bundle,
                           LoudsNativeCluster* cl,
                           std::string* err) {
    if (!cl || !bundle.norm_owned) {
        if (err) *err = "LOUDS-native eval: norm provider missing.";
        return false;
    }
    std::vector<std::uint32_t> parent_1based;
    eval::ClusterView cv;
    if (!BuildProviderClusterView(use_coeff_codec, bundle, *cl, &parent_1based, &cv, err)) {
        return false;
    }
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
    if (auto* cuda_norm = dynamic_cast<eval::cuda::LinkageNormProviderLookupCuda*>(bundle.norm_owned.get())) {
        if (cuda_norm->ComputeNorm2Lut(cv,
                                       cfg.base.encode.hnorms,
                                       cfg.eval.disk_norm2_lut_kmeans_niter,
                                       &cl->norm2_lut,
                                       err)) {
            return true;
        }
    }
#endif
    if (!bundle.norm_owned->ComputeNorm2(cv, &cl->r_norm2, err)) {
        return false;
    }
    return eval::BuildNorm2LutWithMode(cl->r_norm2.data(),
                                       static_cast<int>(cl->r_norm2.size()),
                                       cfg.base.encode.hnorms,
                                       cfg.eval.disk_norm2_lut_kmeans_niter,
                                       cfg.eval.disk_norm2_mode,
                                       cfg.eval.disk_norm2_lut_log_alpha,
                                       cfg.eval.disk_norm2_lut_piecewise_p1,
                                       cfg.eval.disk_norm2_lut_piecewise_p2,
                                       cfg.eval.disk_norm2_lut_piecewise_count_weight,
                                       cfg.eval.disk_norm2_lut_piecewise_range_weight,
                                       &cl->norm2_lut,
                                       err);
}

struct ClusterLoadStats {
    double louds_read_sec = 0.0;
    double louds_deser_sec = 0.0;
    double coeff_io_sec = 0.0;
    double coeff_decode_sec = 0.0;
};

struct ScanScratch {
    std::vector<float> dot_all;
    std::vector<float> root_small_scaled;
    std::vector<float> one_scaled;
    std::vector<float> dist_tmp;
};

void BuildLookupGram(const CodebookMeta& meta_root_small,
                     const CodebookMeta& meta_one,
                     ColMajorMatrix<float>* D_rr,
                     ColMajorMatrix<float>* D_oo,
                     ColMajorMatrix<float>* D_ro) {
    if (D_rr) {
        *D_rr = ColMajorMatrix<float>(meta_root_small.total_cols, meta_root_small.total_cols);
    }
    if (D_oo) {
        *D_oo = ColMajorMatrix<float>(meta_one.total_cols, meta_one.total_cols);
    }
    if (D_ro) {
        *D_ro = ColMajorMatrix<float>(meta_root_small.total_cols, meta_one.total_cols);
    }
    ScopedBlasThreads scope(OmpMaxThreads());
    if (D_rr) Gemm(true, false, 1.0f, meta_root_small.flat, meta_root_small.flat, 0.0f, D_rr);
    if (D_oo) Gemm(true, false, 1.0f, meta_one.flat, meta_one.flat, 0.0f, D_oo);
    if (D_ro) Gemm(true, false, 1.0f, meta_root_small.flat, meta_one.flat, 0.0f, D_ro);
}

int CountRootRealFromLOUDS(const LoudsNativeCluster& cl) {
    int roots = 0;
    auto dec = cl.louds.MakeSequentialParentDecoder();
    dec.Skip(static_cast<std::size_t>(std::max(0, cl.n_virt)));
    for (int pos = 0; pos < cl.n_real; ++pos) {
        if (dec.NextParent1Based() != 0u) break;
        ++roots;
    }
    return roots;
}

enum class Norm2DiskLoadResult {
    kMissing = 0,
    kLoaded = 1,
    kInvalid = 2,
};

Norm2DiskLoadResult LoadNorm2CacheFromDisk(const io::LinkageListReader& linkage_list,
                                          bool use_coeff_codec,
                                          const char* filename,
                                          std::unordered_map<int, std::vector<float>>* out_cache,
                                          double* out_load_sec,
                                          std::string* out_err) {
    if (!out_cache) return Norm2DiskLoadResult::kInvalid;
    const auto path = std::filesystem::path(linkage_list.dir()) / filename;
    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || ec) {
        out_cache->clear();
        return Norm2DiskLoadResult::kMissing;
    }
    std::uint64_t store_hash = 0;
    if (!ReadSourceHashForNorm2Cache(linkage_list, use_coeff_codec, &store_hash)) {
        out_cache->clear();
        return Norm2DiskLoadResult::kMissing;
    }
    std::uint64_t cache_hash = 0;
    const auto cache_hash_path = Norm2CacheHashPath(path);
    if (!app::ReadU64File(cache_hash_path.string(), &cache_hash) || cache_hash != store_hash) {
        out_cache->clear();
        return Norm2DiskLoadResult::kMissing;
    }
    std::uint64_t cache_count = 0;
    if (!app::ReadU64File(Norm2CacheCountPath(path).string(), &cache_count) ||
        cache_count != CountNonEmptyRealClusters(linkage_list)) {
        out_cache->clear();
        return Norm2DiskLoadResult::kMissing;
    }
    const auto& real_offs = linkage_list.real_offsets();
    const int nlist = linkage_list.nlist();
    const std::uint64_t total_real = linkage_list.total_real();
    const std::uint64_t expected_bytes = total_real * sizeof(float);
    const auto actual_bytes = std::filesystem::file_size(path, ec);
    if (ec || actual_bytes != expected_bytes ||
        real_offs.size() != static_cast<std::size_t>(nlist + 1)) {
        if (out_err) {
            *out_err = "LOUDS-native eval: norm2 cache file exists but is invalid: " + path.string();
        }
        return Norm2DiskLoadResult::kInvalid;
    }

    Timer t;
    std::ifstream fin(path, std::ios::binary);
    if (!fin) {
        if (out_err) *out_err = "LOUDS-native eval: failed to open norm2 cache: " + path.string();
        return Norm2DiskLoadResult::kInvalid;
    }
    std::vector<float> flat(static_cast<std::size_t>(total_real));
    fin.read(reinterpret_cast<char*>(flat.data()),
             static_cast<std::streamsize>(flat.size() * sizeof(float)));
    if (!fin) {
        if (out_err) *out_err = "LOUDS-native eval: failed to read norm2 cache: " + path.string();
        return Norm2DiskLoadResult::kInvalid;
    }

    out_cache->clear();
    out_cache->reserve(static_cast<std::size_t>(nlist));
    for (int cid = 0; cid < nlist; ++cid) {
        const auto lo = real_offs[static_cast<std::size_t>(cid)];
        const auto hi = real_offs[static_cast<std::size_t>(cid + 1)];
        const auto nr = hi - lo;
        if (nr == 0) continue;
        std::vector<float> v(static_cast<std::size_t>(nr));
        std::memcpy(v.data(), &flat[static_cast<std::size_t>(lo)],
                    static_cast<std::size_t>(nr) * sizeof(float));
        out_cache->emplace(cid, std::move(v));
    }
    if (out_load_sec) *out_load_sec = t.ElapsedSeconds();
    return Norm2DiskLoadResult::kLoaded;
}

Norm2DiskLoadResult CheckFloatNorm2Cache(const io::LinkageListReader& linkage_list,
                                         bool use_coeff_codec,
                                         const char* filename,
                                         std::string* out_err) {
    const auto path = std::filesystem::path(linkage_list.dir()) / filename;
    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || ec) {
        return Norm2DiskLoadResult::kMissing;
    }
    std::uint64_t store_hash = 0;
    if (!ReadSourceHashForNorm2Cache(linkage_list, use_coeff_codec, &store_hash)) {
        return Norm2DiskLoadResult::kMissing;
    }
    std::uint64_t cache_hash = 0;
    const auto cache_hash_path = Norm2CacheHashPath(path);
    if (!app::ReadU64File(cache_hash_path.string(), &cache_hash) || cache_hash != store_hash) {
        return Norm2DiskLoadResult::kMissing;
    }
    std::uint64_t cache_count = 0;
    if (!app::ReadU64File(Norm2CacheCountPath(path).string(), &cache_count) ||
        cache_count != CountNonEmptyRealClusters(linkage_list)) {
        return Norm2DiskLoadResult::kMissing;
    }
    const std::uint64_t expected_bytes = linkage_list.total_real() * sizeof(float);
    const auto actual_bytes = std::filesystem::file_size(path, ec);
    if (ec || actual_bytes != expected_bytes ||
        linkage_list.real_offsets().size() != static_cast<std::size_t>(linkage_list.nlist() + 1)) {
        if (out_err) {
            *out_err = "LOUDS-native eval: norm2 cache file exists but is invalid: " + path.string();
        }
        return Norm2DiskLoadResult::kInvalid;
    }
    return Norm2DiskLoadResult::kLoaded;
}

bool StoreNorm2CacheToDisk(const io::LinkageListReader& linkage_list,
                          bool use_coeff_codec,
                          const char* filename,
                          const std::unordered_map<int, LoudsNativeCluster>& cluster_cache) {
    const auto path = std::filesystem::path(linkage_list.dir()) / filename;
    std::error_code ec;
    if (std::filesystem::exists(path, ec) && !ec) {
        std::uint64_t store_hash = 0;
        std::uint64_t cache_hash = 0;
        std::uint64_t cache_count = 0;
        if (ReadSourceHashForNorm2Cache(linkage_list, use_coeff_codec, &store_hash) &&
            app::ReadU64File(Norm2CacheHashPath(path).string(), &cache_hash) &&
            app::ReadU64File(Norm2CacheCountPath(path).string(), &cache_count) &&
            cache_count == CountNonEmptyRealClusters(linkage_list) &&
            cache_hash == store_hash) {
            return true;
        }
    }
    const auto& real_offs = linkage_list.real_offsets();
    const int nlist = linkage_list.nlist();
    const std::uint64_t total_real = linkage_list.total_real();
    if (real_offs.size() != static_cast<std::size_t>(nlist + 1) || total_real == 0) return true;

    std::vector<float> flat(static_cast<std::size_t>(total_real), 0.0f);
    int stored = 0;
    for (const auto& [cid, cl] : cluster_cache) {
        if (cid < 0 || cid >= nlist) continue;
        const auto lo = real_offs[static_cast<std::size_t>(cid)];
        const auto hi = real_offs[static_cast<std::size_t>(cid + 1)];
        const auto nr = hi - lo;
        if (cl.r_norm2.size() != static_cast<std::size_t>(nr)) continue;
        std::memcpy(&flat[static_cast<std::size_t>(lo)], cl.r_norm2.data(),
                    static_cast<std::size_t>(nr) * sizeof(float));
        ++stored;
    }
    const std::uint64_t expected_clusters = CountNonEmptyRealClusters(linkage_list);
    if (stored <= 0 || static_cast<std::uint64_t>(stored) != expected_clusters) return true;
    std::ofstream fout(path, std::ios::binary);
    if (!fout) return true;
    fout.write(reinterpret_cast<const char*>(flat.data()),
               static_cast<std::streamsize>(flat.size() * sizeof(float)));
    if (!fout) return false;
    std::uint64_t store_hash = 0;
    if (ReadSourceHashForNorm2Cache(linkage_list, use_coeff_codec, &store_hash)) {
        std::string hash_err;
        (void)app::WriteU64FileHex(Norm2CacheHashPath(path).string(), store_hash, &hash_err);
        std::string count_err;
        (void)app::WriteU64FileHex(Norm2CacheCountPath(path).string(), expected_clusters, &count_err);
    }
    return true;
}

bool StoreNorm2CacheVectorsToDisk(const io::LinkageListReader& linkage_list,
                                  bool use_coeff_codec,
                                  const std::filesystem::path& path,
                                  const std::unordered_map<int, std::vector<float>>& norm_cache,
                                  std::string* err) {
    const auto& real_offs = linkage_list.real_offsets();
    const int nlist = linkage_list.nlist();
    const std::uint64_t total_real = linkage_list.total_real();
    if (real_offs.size() != static_cast<std::size_t>(nlist + 1) || total_real == 0) {
        return true;
    }

    std::vector<float> flat(static_cast<std::size_t>(total_real), 0.0f);
    int stored = 0;
    for (const auto& [cid, r_norm2] : norm_cache) {
        if (cid < 0 || cid >= nlist) continue;
        const auto lo = real_offs[static_cast<std::size_t>(cid)];
        const auto hi = real_offs[static_cast<std::size_t>(cid + 1)];
        const auto nr = hi - lo;
        if (r_norm2.size() != static_cast<std::size_t>(nr)) continue;
        std::memcpy(&flat[static_cast<std::size_t>(lo)], r_norm2.data(),
                    static_cast<std::size_t>(nr) * sizeof(float));
        ++stored;
    }

    const std::uint64_t expected_clusters = CountNonEmptyRealClusters(linkage_list);
    if (stored <= 0 || static_cast<std::uint64_t>(stored) != expected_clusters) {
        if (err) {
            *err = "StoreNorm2CacheVectorsToDisk: incomplete cluster coverage (" +
                   std::to_string(stored) + "/" + std::to_string(expected_clusters) + ").";
        }
        return false;
    }

    std::ofstream fout(path, std::ios::binary);
    if (!fout) {
        if (err) *err = "StoreNorm2CacheVectorsToDisk: failed to open " + path.string();
        return false;
    }
    fout.write(reinterpret_cast<const char*>(flat.data()),
               static_cast<std::streamsize>(flat.size() * sizeof(float)));
    if (!fout) {
        if (err) *err = "StoreNorm2CacheVectorsToDisk: failed to write " + path.string();
        return false;
    }
    std::uint64_t store_hash = 0;
    if (ReadSourceHashForNorm2Cache(linkage_list, use_coeff_codec, &store_hash)) {
        std::string hash_err;
        if (!app::WriteU64FileHex(Norm2CacheHashPath(path).string(), store_hash, &hash_err)) {
            if (err) *err = hash_err;
            return false;
        }
        std::string count_err;
        if (!app::WriteU64FileHex(Norm2CacheCountPath(path).string(), expected_clusters, &count_err)) {
            if (err) *err = count_err;
            return false;
        }
    }
    return true;
}

bool ComputeNorm2Louds(bool use_coeff_codec,const LoudsNativeBundle& bundle,
                       LoudsNativeCluster* cl,
                       std::string* err) {
    if (!cl || !bundle.C_root0) return false;
    const int d = bundle.meta_one.d;
    const int m = cl->m;
    const int m_codes = cl->m_codes;
    if (d <= 0 || m <= 1 || m > kMaxSupportedModelM || m_codes != m - 1) {
        if (err) *err = "LOUDS-native eval: invalid dims for norm2.";
        return false;
    }
    if (cl->cid < 0 || cl->cid >= bundle.C_root0->cols) {
        if (err) *err = "LOUDS-native eval: invalid cid for norm2.";
        return false;
    }
    if (cl->n_real <= 0) {
        cl->r_norm2.clear();
        return true;
    }

    const int n_virt = cl->n_virt;
    const int n_real = cl->n_real;
    [[maybe_unused]] const int nc = cl->nc;
    const int real_base = n_virt;
    const int n_root_real = cl->n_root_real;
    const int total_nodes = n_virt + n_real;
    if (total_nodes <= 0) {
        cl->r_norm2.clear();
        return true;
    }

    // NOTE: norm2 reconstruction requires ancestor walks. We build a temporary parent table
    // from the sequential LOUDS decoder (no rank/select queries; not persisted).
    std::vector<int> parent0(static_cast<std::size_t>(total_nodes), -1);
    {
        auto dec = cl->louds.MakeSequentialParentDecoder();
        for (int node = 0; node < total_nodes; ++node) {
            parent0[static_cast<std::size_t>(node)] = static_cast<int>(dec.NextParent1Based()) - 1;
        }
    }

    const float* c0 = bundle.C_root0->Col(cl->cid);
    const float c0_norm2 = Norm2(c0, d);

    std::vector<float> c0_dot_rootS(static_cast<std::size_t>(bundle.meta_root_small.total_cols), 0.0f);
    for (int j = 0; j < bundle.meta_root_small.total_cols; ++j) {
        c0_dot_rootS[static_cast<std::size_t>(j)] = Dot(c0, bundle.meta_root_small.flat.Col(j), d);
    }
    std::vector<float> c0_dot_one(static_cast<std::size_t>(bundle.meta_one.total_cols), 0.0f);
    for (int j = 0; j < bundle.meta_one.total_cols; ++j) {
        c0_dot_one[static_cast<std::size_t>(j)] = Dot(c0, bundle.meta_one.flat.Col(j), d);
    }

    std::vector<float> norm2_real(static_cast<std::size_t>(n_real), 0.0f);
    std::vector<float> norm2_virt(static_cast<std::size_t>(std::max(0, n_virt)), 0.0f);
    std::vector<int> offsets_root_small(static_cast<std::size_t>(m), 0);
    for (int l = 1; l < m; ++l) {
        offsets_root_small[static_cast<std::size_t>(l)] =
            bundle.meta_root_small.offsets[static_cast<std::size_t>(l - 1)];
    }

    const std::uint8_t* codes = cl->codes_small.data();
    const std::uint8_t* code0_one = cl->code0_one.data();
    const std::uint8_t* vcodes = cl->virt_codes_small.data();

    const auto compute_root_norm2 = [&](const std::uint8_t* codes_src, std::size_t pos_src,
                                        float a0, auto a_layer_fn) -> float {
        int idx[kMaxSupportedModelM];
        float a[kMaxSupportedModelM];
        for (int l = 1; l < m; ++l) {
            const int code = ReadSmallCode(codes_src, static_cast<int>(pos_src), m_codes, l - 1);
            idx[l - 1] = offsets_root_small[static_cast<std::size_t>(l)] + code;
            a[l - 1] = a_layer_fn(l);
        }
        float cross0 = 0.0f;
        for (int l = 1; l < m; ++l) {
            cross0 += a[l - 1] * c0_dot_rootS[static_cast<std::size_t>(idx[l - 1])];
        }
        float small = 0.0f;
        for (int j = 1; j < m; ++j) {
            const float aj = a[j - 1];
            const int fj = idx[j - 1];
            small += (aj * aj) * bundle.D_rr(fj, fj);
            for (int k = j + 1; k < m; ++k) {
                small += (2.0f * aj * a[k - 1]) * bundle.D_rr(fj, idx[k - 1]);
            }
        }
        return a0 * a0 * c0_norm2 + 2.0f * a0 * cross0 + small;
    };

    for (int v = 0; v < n_virt; ++v) {
        const auto vp = static_cast<std::size_t>(v);
        float a0 = 0.0f;
        if (use_coeff_codec) {
            a0 = cl->scales_root[0] * static_cast<float>(cl->q_layer_major[static_cast<std::size_t>(v)]);
            norm2_virt[vp] = compute_root_norm2(vcodes, vp, a0, [&](int l) {
                const std::size_t qidx = static_cast<std::size_t>(l) * static_cast<std::size_t>(nc) +
                                         static_cast<std::size_t>(v);
                return cl->scales_root[static_cast<std::size_t>(l)] *
                       static_cast<float>(cl->q_layer_major[qidx]);
            });
        } else {
            a0 = cl->virt_a0[vp];
            norm2_virt[vp] = compute_root_norm2(vcodes, vp, a0, [&](int l) {
                return cl->virt_coeffs_small[vp * static_cast<std::size_t>(m_codes) +
                                             static_cast<std::size_t>(l - 1)];
            });
        }
    }

    for (int pos = 0; pos < n_root_real; ++pos) {
        const auto p = static_cast<std::size_t>(pos);
        [[maybe_unused]] const int local = real_base + pos;
        float a0 = 0.0f;
        if (use_coeff_codec) {
            a0 = cl->scales_root[0] * static_cast<float>(cl->q_layer_major[static_cast<std::size_t>(local)]);
            norm2_real[p] = compute_root_norm2(codes, p, a0, [&](int l) {
                const std::size_t qidx = static_cast<std::size_t>(l) * static_cast<std::size_t>(nc) +
                                         static_cast<std::size_t>(local);
                return cl->scales_root[static_cast<std::size_t>(l)] *
                       static_cast<float>(cl->q_layer_major[qidx]);
            });
        } else {
            a0 = cl->a0[p];
            norm2_real[p] = compute_root_norm2(codes, p, a0, [&](int l) {
                return cl->coeffs_small[p * static_cast<std::size_t>(m_codes) +
                                        static_cast<std::size_t>(l - 1)];
            });
        }
    }

    int idx_i[kMaxSupportedModelM];
    float b_i[kMaxSupportedModelM];
    int idx_u[kMaxSupportedModelM];
    float b_u[kMaxSupportedModelM];
    for (int pos = n_root_real; pos < n_real; ++pos) {
        const auto ppos = static_cast<std::size_t>(pos);
        const int local_ppos = real_base + pos;
        const int p = (local_ppos >= 0 && local_ppos < total_nodes)
            ? parent0[static_cast<std::size_t>(local_ppos)]
            : -1;

        idx_i[0] = bundle.meta_one.offsets[0] + static_cast<int>(code0_one[static_cast<std::size_t>(pos)]);
        if (use_coeff_codec) {
            b_i[0] = cl->scales_linkage[0] * static_cast<float>(cl->q_layer_major[static_cast<std::size_t>(local_ppos)]);
        } else {
            b_i[0] = cl->a0[ppos];
        }
        for (int l = 1; l < m; ++l) {
            const int code = static_cast<int>(ReadSmallCode(codes, pos, m_codes, l - 1));
            idx_i[l] = bundle.meta_one.offsets[static_cast<std::size_t>(l)] + code;
            if (use_coeff_codec) {
                const std::size_t qidx = static_cast<std::size_t>(l) * static_cast<std::size_t>(nc) +
                                         static_cast<std::size_t>(local_ppos);
                b_i[l] = cl->scales_linkage[static_cast<std::size_t>(l)] *
                         static_cast<float>(cl->q_layer_major[qidx]);
            } else {
                b_i[l] = cl->coeffs_small[ppos * static_cast<std::size_t>(m_codes) +
                                          static_cast<std::size_t>(l - 1)];
            }
        }

        float res_norm2 = 0.0f;
        for (int j = 0; j < m; ++j) {
            const float aj = b_i[j];
            const int fj = idx_i[j];
            res_norm2 += (aj * aj) * bundle.D_oo(fj, fj);
            for (int k = j + 1; k < m; ++k) {
                res_norm2 += (2.0f * aj * b_i[k]) * bundle.D_oo(fj, idx_i[k]);
            }
        }

        float cross = 0.0f;
        int a = p;
        while (a >= 0) {
            if (a < real_base) {
                const int v = a;
                const auto vp = static_cast<std::size_t>(v);
                float a0 = 0.0f;
                if (use_coeff_codec) {
                    a0 = cl->scales_root[0] * static_cast<float>(cl->q_layer_major[static_cast<std::size_t>(v)]);
                } else {
                    a0 = cl->virt_a0[vp];
                }
                float dot0 = 0.0f;
                for (int k = 0; k < m; ++k) {
                    dot0 += b_i[k] * c0_dot_one[static_cast<std::size_t>(idx_i[k])];
                }
                cross += a0 * dot0;
                for (int l = 1; l < m; ++l) {
                    const int code = static_cast<int>(ReadSmallCode(vcodes, v, m_codes, l - 1));
                    const int idx_rs = offsets_root_small[static_cast<std::size_t>(l)] + code;
                    float al = 0.0f;
                    if (use_coeff_codec) {
                        const std::size_t qidx = static_cast<std::size_t>(l) * static_cast<std::size_t>(nc) +
                                                 static_cast<std::size_t>(v);
                        al = cl->scales_root[static_cast<std::size_t>(l)] *
                             static_cast<float>(cl->q_layer_major[qidx]);
                    } else {
                        al = cl->virt_coeffs_small[vp * static_cast<std::size_t>(m_codes) +
                                                   static_cast<std::size_t>(l - 1)];
                    }
                    float acc = 0.0f;
                    for (int k = 0; k < m; ++k) acc += b_i[k] * bundle.D_ro(idx_rs, idx_i[k]);
                    cross += al * acc;
                }
                break;
            }
            const int a_real = a - real_base;
            if (a_real >= 0 && a_real < n_root_real) {
                const auto rp = static_cast<std::size_t>(a_real);
                float a0 = 0.0f;
                if (use_coeff_codec) {
                    a0 = cl->scales_root[0] * static_cast<float>(cl->q_layer_major[static_cast<std::size_t>(a)]);
                } else {
                    a0 = cl->a0[rp];
                }
                float dot0 = 0.0f;
                for (int k = 0; k < m; ++k) dot0 += b_i[k] * c0_dot_one[static_cast<std::size_t>(idx_i[k])];
                cross += a0 * dot0;
                for (int l = 1; l < m; ++l) {
                    const int code = static_cast<int>(ReadSmallCode(codes, a_real, m_codes, l - 1));
                    const int idx_rs = offsets_root_small[static_cast<std::size_t>(l)] + code;
                    float al = 0.0f;
                    if (use_coeff_codec) {
                        const std::size_t qidx = static_cast<std::size_t>(l) * static_cast<std::size_t>(nc) +
                                                 static_cast<std::size_t>(a);
                        al = cl->scales_root[static_cast<std::size_t>(l)] *
                             static_cast<float>(cl->q_layer_major[qidx]);
                    } else {
                        al = cl->coeffs_small[rp * static_cast<std::size_t>(m_codes) +
                                              static_cast<std::size_t>(l - 1)];
                    }
                    float acc = 0.0f;
                    for (int k = 0; k < m; ++k) acc += b_i[k] * bundle.D_ro(idx_rs, idx_i[k]);
                    cross += al * acc;
                }
                break;
            }
            if (a_real < 0 || a_real >= n_real) break;
            const auto up = static_cast<std::size_t>(a_real);
            idx_u[0] = bundle.meta_one.offsets[0] + static_cast<int>(code0_one[static_cast<std::size_t>(a_real)]);
            if (use_coeff_codec) {
                b_u[0] = cl->scales_linkage[0] * static_cast<float>(cl->q_layer_major[static_cast<std::size_t>(a)]);
            } else {
                b_u[0] = cl->a0[up];
            }
            for (int l = 1; l < m; ++l) {
                const int code = static_cast<int>(ReadSmallCode(codes, a_real, m_codes, l - 1));
                idx_u[l] = bundle.meta_one.offsets[static_cast<std::size_t>(l)] + code;
                if (use_coeff_codec) {
                    const std::size_t qidx = static_cast<std::size_t>(l) * static_cast<std::size_t>(nc) +
                                             static_cast<std::size_t>(a);
                    b_u[l] = cl->scales_linkage[static_cast<std::size_t>(l)] *
                             static_cast<float>(cl->q_layer_major[qidx]);
                } else {
                    b_u[l] = cl->coeffs_small[up * static_cast<std::size_t>(m_codes) +
                                              static_cast<std::size_t>(l - 1)];
                }
            }
            float dot = 0.0f;
            for (int j = 0; j < m; ++j) {
                const float aj = b_u[j];
                const int fj = idx_u[j];
                for (int k = 0; k < m; ++k) dot += (aj * b_i[k]) * bundle.D_oo(fj, idx_i[k]);
            }
            cross += dot;
            if (a < 0 || a >= total_nodes) break;
            a = parent0[static_cast<std::size_t>(a)];
        }

        float parent_norm = 0.0f;
        if (p >= 0) {
            parent_norm = (p < real_base)
                ? norm2_virt[static_cast<std::size_t>(p)]
                : norm2_real[static_cast<std::size_t>(p - real_base)];
        }
        norm2_real[ppos] = parent_norm + res_norm2 + 2.0f * cross;
    }

    cl->r_norm2 = std::move(norm2_real);
    return true;
}

bool OpenBundle(bool use_coeff_codec,const Config& cfg,
                const io::LinkageListReader& linkage_list,
                const TrainResult& train,
                LoudsNativeBundle* bundle,
                std::string* err) {
    if (!bundle) return false;
    bundle->linkage_list = &linkage_list;
    bundle->use_coeff_codec = use_coeff_codec;
    bundle->C_root0 = &train.C_root.books.front();
    if (!bundle->linkage_thr.OpenFrom(linkage_list, /*require_coeffs_f32=*/!use_coeff_codec, err)) {
        return false;
    }
    if (use_coeff_codec) {
        std::string local_err;
        if (!bundle->coeff_reader.Open(linkage_list.dir(), &local_err)) {
            if (err) *err = local_err.empty() ? "LOUDS-native eval: missing coeff codec store." : local_err;
            return false;
        }
        if (!bundle->coeff_thr.OpenFrom(bundle->coeff_reader, &local_err)) {
            if (err) *err = local_err;
            return false;
        }
    }
    std::vector<const ColMajorMatrix<float>*> root_small_books;
    root_small_books.reserve(static_cast<std::size_t>(cfg.model.m - 1));
    for (int l = 1; l < cfg.model.m; ++l) {
        root_small_books.push_back(&train.C_root.books[static_cast<std::size_t>(l)]);
    }
    bundle->meta_root_small = BuildCodebookMeta(root_small_books);
    bundle->meta_one = BuildCodebookMeta(GatherBooks(train.C_one));
    BuildLookupGram(bundle->meta_root_small, bundle->meta_one, &bundle->D_rr, &bundle->D_oo, &bundle->D_ro);
    bundle->norm_owned = MakeNormProviderOwnedNative(
        cfg, *bundle->C_root0, bundle->meta_root_small, bundle->meta_one);
    bundle->cluster_cache.reserve(static_cast<std::size_t>(linkage_list.nlist()));
    const bool use_norm2_lut = eval::IsDiskNorm2ModeLut(cfg.eval.disk_norm2_mode);
    bundle->norm2_lut_global = eval::IsDiskNorm2ModeLutGlobal(cfg.eval.disk_norm2_mode);
    const bool is_lazy_mode = (cfg.eval.linkage_preload_clusters_io_threads < 0);
    if (use_norm2_lut) {
        std::string lut_err;
        const auto lut_res = bundle->norm2_lut_global
            ? eval::CheckNorm2LutCache(linkage_list, use_coeff_codec,
                                       cfg.base.encode.hnorms,
                                       cfg.eval.disk_norm2_lut_kmeans_niter,
                                       cfg.eval.disk_norm2_mode,
                                       cfg.eval.disk_norm2_lut_log_alpha,
                                       cfg.eval.disk_norm2_lut_piecewise_p1,
                                       cfg.eval.disk_norm2_lut_piecewise_p2,
                                       cfg.eval.disk_norm2_lut_piecewise_count_weight,
                                       cfg.eval.disk_norm2_lut_piecewise_range_weight,
                                       &lut_err)
            : (is_lazy_mode
                ? eval::CheckNorm2LutClusterCache(linkage_list, use_coeff_codec,
                                                  cfg.base.encode.hnorms,
                                                  cfg.eval.disk_norm2_lut_kmeans_niter,
                                                  &lut_err)
                : eval::LoadNorm2LutClusterCache(linkage_list, use_coeff_codec,
                                                 cfg.base.encode.hnorms,
                                                 cfg.eval.disk_norm2_lut_kmeans_niter,
                                                 &bundle->norm2_lut_cache, &lut_err));
        bundle->norm2_lut_disk_present = (lut_res == eval::Norm2LutDiskLoadResult::kLoaded);
        bundle->norm2_lut_disk_lazy_enabled = bundle->norm2_lut_disk_present && is_lazy_mode;
        if (lut_res == eval::Norm2LutDiskLoadResult::kInvalid) {
            LogWarn("LOUDS-native eval: ignoring stale norm2 LUT cache (" +
                    (lut_err.empty() ? std::string("invalid cache") : lut_err) + ").");
            bundle->norm2_lut_cache.clear();
            bundle->global_norm2_lut_centers.clear();
            bundle->norm2_lut_disk_present = false;
            bundle->norm2_lut_disk_lazy_enabled = false;
        } else if (bundle->norm2_lut_global && bundle->norm2_lut_disk_present) {
            if (!eval::LoadNorm2LutCenters(linkage_list, use_coeff_codec,
                                           cfg.base.encode.hnorms,
                                           cfg.eval.disk_norm2_lut_kmeans_niter,
                                           cfg.eval.disk_norm2_mode,
                                           cfg.eval.disk_norm2_lut_log_alpha,
                                           cfg.eval.disk_norm2_lut_piecewise_p1,
                                           cfg.eval.disk_norm2_lut_piecewise_p2,
                                           cfg.eval.disk_norm2_lut_piecewise_count_weight,
                                           cfg.eval.disk_norm2_lut_piecewise_range_weight,
                                           &bundle->global_norm2_lut_centers, &lut_err)) {
                if (err) *err = "LOUDS-native eval: failed to load global norm2 LUT centers: " + lut_err;
                return false;
            }
            if (!is_lazy_mode) {
                eval::Norm2Lut preloaded_lut;
                if (eval::LoadNorm2LutCache(linkage_list, use_coeff_codec,
                                            cfg.base.encode.hnorms,
                                            cfg.eval.disk_norm2_lut_kmeans_niter,
                                            cfg.eval.disk_norm2_mode,
                                            cfg.eval.disk_norm2_lut_log_alpha,
                                            cfg.eval.disk_norm2_lut_piecewise_p1,
                                            cfg.eval.disk_norm2_lut_piecewise_p2,
                                            cfg.eval.disk_norm2_lut_piecewise_count_weight,
                                            cfg.eval.disk_norm2_lut_piecewise_range_weight,
                                            &preloaded_lut, &lut_err) !=
                    eval::Norm2LutDiskLoadResult::kLoaded) {
                    if (err) *err = "LOUDS-native eval: failed to load global norm2 LUT cache: " + lut_err;
                    return false;
                }
                const auto& real_offs = linkage_list.real_offsets();
                for (int cid = 0; cid < linkage_list.nlist(); ++cid) {
                    const auto lo = real_offs[static_cast<std::size_t>(cid)];
                    const auto hi = real_offs[static_cast<std::size_t>(cid + 1)];
                    const auto nr = hi - lo;
                    if (nr == 0) continue;
                    eval::Norm2Lut lut;
                    lut.codes_u8.resize(static_cast<std::size_t>(nr));
                    std::memcpy(lut.codes_u8.data(),
                                preloaded_lut.codes_u8.data() + static_cast<std::size_t>(lo),
                                static_cast<std::size_t>(nr));
                    bundle->norm2_lut_cache.emplace(cid, std::move(lut));
                }
            }
        }
    } else {
        const char* norm2_filename = use_coeff_codec ? "norm2_int8.f32" : "norm2_float.f32";
        std::string norm2_err;
        const auto norm2_res = is_lazy_mode
            ? CheckFloatNorm2Cache(linkage_list, use_coeff_codec, norm2_filename, &norm2_err)
            : LoadNorm2CacheFromDisk(linkage_list,
                                     use_coeff_codec,
                                     norm2_filename,
                                     &bundle->norm2_cache,
                                     nullptr,
                                     &norm2_err);
        bundle->norm2_disk_present = (norm2_res == Norm2DiskLoadResult::kLoaded);
        bundle->norm2_disk_lazy_enabled = bundle->norm2_disk_present && is_lazy_mode;
        if (norm2_res == Norm2DiskLoadResult::kInvalid) {
            LogWarn("LOUDS-native eval: ignoring stale norm2 cache (" +
                    (norm2_err.empty() ? std::string("invalid cache") : norm2_err) + ").");
            bundle->norm2_cache.clear();
            bundle->norm2_disk_present = false;
            bundle->norm2_disk_lazy_enabled = false;
        }
    }
    return true;
}

bool LoadClusterWithReaders(bool use_coeff_codec,LoudsNativeBundle* bundle,
                            io::LinkageListThreadReader& linkage_thr,
                            io::LinkageCoeffCodecThreadReader* coeff_thr,
                            int cid,
                            const Config& cfg,
                            LoudsNativeCluster* cl,
                            ClusterLoadStats* stats,
                            std::string* err);

bool BuildAndStoreFullNorm2CacheNative(bool use_coeff_codec,
                                       const Config& cfg,
                                       const io::LinkageListReader& linkage_list,
                                       const TrainResult& train,
                                       std::string* err) {
    LoudsNativeBundle bundle;
    if (!OpenBundle(use_coeff_codec, cfg, linkage_list, train, &bundle, err)) return false;

    std::unordered_map<int, std::vector<float>> full_norm_cache;
    full_norm_cache.reserve(static_cast<std::size_t>(linkage_list.nlist()));
    for (int cid = 0; cid < linkage_list.nlist(); ++cid) {
        LoudsNativeCluster cl;
        ClusterLoadStats stats{};
        if (!LoadClusterWithReaders(use_coeff_codec,
                                    &bundle,
                                    bundle.linkage_thr,
                                    use_coeff_codec ? &bundle.coeff_thr : nullptr,
                                    cid,
                                    cfg,
                                    &cl,
                                    &stats,
                                    err)) {
            return false;
        }
        if (cl.n_real > 0) {
            full_norm_cache.emplace(cid, std::move(cl.r_norm2));
        }
    }

    const std::filesystem::path out_path =
        std::filesystem::path(linkage_list.dir()) /
        (use_coeff_codec ? "norm2_int8.f32" : "norm2_float.f32");
    return StoreNorm2CacheVectorsToDisk(linkage_list, use_coeff_codec, out_path, full_norm_cache, err);
}

bool BuildAndStoreFullNorm2LutCacheNative(bool use_coeff_codec,
                                          const Config& cfg,
                                          const io::LinkageListReader& linkage_list,
                                          const TrainResult& train,
                                          std::string* err) {
    LoudsNativeBundle bundle;
    if (!OpenBundle(use_coeff_codec, cfg, linkage_list, train, &bundle, err)) return false;

    const bool use_global_lut = eval::IsDiskNorm2ModeLutGlobal(cfg.eval.disk_norm2_mode);
    Config build_cfg = cfg;
    if (use_global_lut) {
        build_cfg.eval.disk_norm2_mode = "float";
    }
    std::unordered_map<int, eval::Norm2Lut> full_lut_cache;
    std::unordered_map<int, std::vector<float>> full_norm_cache;
    const char* norm2_filename = use_coeff_codec ? "norm2_int8.f32" : "norm2_float.f32";
    std::string norm2_err;
    if (LoadNorm2CacheFromDisk(linkage_list,
                               use_coeff_codec,
                               norm2_filename,
                               &full_norm_cache,
                               nullptr,
                               &norm2_err) == Norm2DiskLoadResult::kLoaded) {
        LogInfo("Reusing validated float norm2 cache " + std::string(norm2_filename) +
                " to build " + std::string(use_global_lut ? "global" : "cluster") + " norm2 LUT cache.");
        if (use_global_lut) {
            const auto& real_offs = linkage_list.real_offsets();
            std::vector<float> flat_norm2(static_cast<std::size_t>(linkage_list.total_real()), 0.0f);
            for (const auto& [cid, r_norm2] : full_norm_cache) {
                const auto lo = real_offs[static_cast<std::size_t>(cid)];
                const auto hi = real_offs[static_cast<std::size_t>(cid + 1)];
                const auto nr = hi - lo;
                if (r_norm2.size() != static_cast<std::size_t>(nr)) continue;
                std::memcpy(flat_norm2.data() + static_cast<std::size_t>(lo),
                            r_norm2.data(),
                            static_cast<std::size_t>(nr) * sizeof(float));
            }
            eval::Norm2Lut global_lut;
            if (!eval::BuildNorm2LutWithMode(flat_norm2.data(),
                                             static_cast<int>(flat_norm2.size()),
                                             cfg.base.encode.hnorms,
                                             cfg.eval.disk_norm2_lut_kmeans_niter,
                                             cfg.eval.disk_norm2_mode,
                                             cfg.eval.disk_norm2_lut_log_alpha,
                                             cfg.eval.disk_norm2_lut_piecewise_p1,
                                             cfg.eval.disk_norm2_lut_piecewise_p2,
                                             cfg.eval.disk_norm2_lut_piecewise_count_weight,
                                             cfg.eval.disk_norm2_lut_piecewise_range_weight,
                                             &global_lut,
                                             err)) {
                return false;
            }
            return eval::StoreNorm2LutCache(linkage_list, use_coeff_codec, cfg.base.encode.hnorms,
                                            cfg.eval.disk_norm2_lut_kmeans_niter,
                                            cfg.eval.disk_norm2_mode,
                                            cfg.eval.disk_norm2_lut_log_alpha,
                                            cfg.eval.disk_norm2_lut_piecewise_p1,
                                            cfg.eval.disk_norm2_lut_piecewise_p2,
                                            cfg.eval.disk_norm2_lut_piecewise_count_weight,
                                            cfg.eval.disk_norm2_lut_piecewise_range_weight,
                                            global_lut, err);
        }
        full_lut_cache.reserve(full_norm_cache.size());
        for (const auto& [cid, r_norm2] : full_norm_cache) {
            eval::Norm2Lut lut;
            if (!bundle.norm_owned->ComputeNorm2LutFromHost(r_norm2.data(),
                                                            static_cast<int>(r_norm2.size()),
                                                            cfg.base.encode.hnorms,
                                                            cfg.eval.disk_norm2_lut_kmeans_niter,
                                                            &lut,
                                                            err)) {
                return false;
            }
            full_lut_cache.emplace(cid, std::move(lut));
        }
        return eval::StoreNorm2LutClusterCache(linkage_list, use_coeff_codec, cfg.base.encode.hnorms,
                                               cfg.eval.disk_norm2_lut_kmeans_niter,
                                               full_lut_cache, err);
    }
    if (!use_global_lut) {
        full_lut_cache.reserve(static_cast<std::size_t>(linkage_list.nlist()));
    } else {
        full_norm_cache.reserve(static_cast<std::size_t>(linkage_list.nlist()));
    }
    for (int cid = 0; cid < linkage_list.nlist(); ++cid) {
        LoudsNativeCluster cl;
        ClusterLoadStats stats{};
        if (!LoadClusterWithReaders(use_coeff_codec,
                                    &bundle,
                                    bundle.linkage_thr,
                                    use_coeff_codec ? &bundle.coeff_thr : nullptr,
                                    cid,
                                    build_cfg,
                                    &cl,
                                    &stats,
                                    err)) {
            return false;
        }
        if (cl.n_real > 0) {
            if (use_global_lut) {
                std::vector<std::uint32_t> parent_1based;
                eval::ClusterView cv;
                if (!BuildProviderClusterView(use_coeff_codec, bundle, cl, &parent_1based, &cv, err)) {
                    return false;
                }
                std::vector<float> r_norm2;
                if (!bundle.norm_owned->ComputeNorm2(cv, &r_norm2, err)) {
                    return false;
                }
                full_norm_cache.emplace(cid, std::move(r_norm2));
            } else {
                full_lut_cache.emplace(cid, std::move(cl.norm2_lut));
            }
        }
    }

    if (use_global_lut) {
        const auto& real_offs = linkage_list.real_offsets();
        std::vector<float> flat_norm2(static_cast<std::size_t>(linkage_list.total_real()), 0.0f);
        for (const auto& [cid, r_norm2] : full_norm_cache) {
            const auto lo = real_offs[static_cast<std::size_t>(cid)];
            const auto hi = real_offs[static_cast<std::size_t>(cid + 1)];
            const auto nr = hi - lo;
            if (r_norm2.size() != static_cast<std::size_t>(nr)) continue;
            std::memcpy(flat_norm2.data() + static_cast<std::size_t>(lo),
                        r_norm2.data(),
                        static_cast<std::size_t>(nr) * sizeof(float));
        }
        eval::Norm2Lut global_lut;
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
        if (auto* cuda_norm = dynamic_cast<eval::cuda::LinkageNormProviderLookupCuda*>(bundle.norm_owned.get())) {
            (void)cuda_norm;
        }
#endif
        if (!eval::BuildNorm2LutWithMode(flat_norm2.data(),
                                         static_cast<int>(flat_norm2.size()),
                                         cfg.base.encode.hnorms,
                                         cfg.eval.disk_norm2_lut_kmeans_niter,
                                         cfg.eval.disk_norm2_mode,
                                         cfg.eval.disk_norm2_lut_log_alpha,
                                         cfg.eval.disk_norm2_lut_piecewise_p1,
                                         cfg.eval.disk_norm2_lut_piecewise_p2,
                                         cfg.eval.disk_norm2_lut_piecewise_count_weight,
                                         cfg.eval.disk_norm2_lut_piecewise_range_weight,
                                         &global_lut,
                                         err)) {
            return false;
        }
        return eval::StoreNorm2LutCache(linkage_list, use_coeff_codec, cfg.base.encode.hnorms,
                                        cfg.eval.disk_norm2_lut_kmeans_niter,
                                        cfg.eval.disk_norm2_mode,
                                        cfg.eval.disk_norm2_lut_log_alpha,
                                        cfg.eval.disk_norm2_lut_piecewise_p1,
                                        cfg.eval.disk_norm2_lut_piecewise_p2,
                                        cfg.eval.disk_norm2_lut_piecewise_count_weight,
                                        cfg.eval.disk_norm2_lut_piecewise_range_weight,
                                        global_lut, err);
    }

    return eval::StoreNorm2LutClusterCache(linkage_list, use_coeff_codec, cfg.base.encode.hnorms,
                                           cfg.eval.disk_norm2_lut_kmeans_niter,
                                           full_lut_cache, err);
}

bool LoadClusterWithReaders(bool use_coeff_codec,LoudsNativeBundle* bundle,
                            io::LinkageListThreadReader& linkage_thr,
                            io::LinkageCoeffCodecThreadReader* coeff_thr,
                            int cid,
                            const Config& cfg,
                            LoudsNativeCluster* cl,
                            ClusterLoadStats* stats,
                            std::string* err) {
    if (!bundle || !cl) return false;
    cl->cid = cid;
    cl->m = cfg.model.m;
    cl->m_codes = std::max(0, cl->m - 1);
    if (stats) *stats = ClusterLoadStats{};

    std::string local_err;
    {
        Timer t;
        if (!linkage_thr.ReadClusterParentLOUDSBlob(cid, &cl->louds_blob, &local_err) || cl->louds_blob.empty()) {
            if (err) *err = local_err.empty() ? ("LOUDS-native eval: missing parent LOUDS blob cid=" + std::to_string(cid)) : local_err;
            return false;
        }
        if (stats) stats->louds_read_sec = t.ElapsedSeconds();
    }
    {
        Timer t;
        cl->louds.Deserialize(cl->louds_blob.data(),
                              cl->louds_blob.size(),
                              static_cast<std::uint32_t>(std::max(1, cfg.eval.parent_louds_rank_words_per_super_log2)),
                              /*build_indices=*/cfg.eval.parent_louds_build_indices,
                              static_cast<std::uint32_t>(std::max(1, cfg.eval.parent_louds_select_stride)));
        if (stats) stats->louds_deser_sec = t.ElapsedSeconds();
    }
    if (!linkage_thr.ReadCluster(cid,
                               &cl->real_ids,
                               nullptr,
                               &cl->depth_offsets,
                               &cl->codes_small,
                               use_coeff_codec ? nullptr : &cl->coeffs_small,
                               &cl->code0_one,
                               use_coeff_codec ? nullptr : &cl->a0,
                               &cl->virt_codes_small,
                               use_coeff_codec ? nullptr : &cl->virt_coeffs_small,
                               use_coeff_codec ? nullptr : &cl->virt_a0,
                               &local_err)) {
        if (err) *err = local_err.empty() ? ("LOUDS-native eval: failed to read cluster cid=" + std::to_string(cid)) : local_err;
        return false;
    }

    cl->n_real = static_cast<int>(cl->real_ids.size());
    // Eval contract: small-layer codes are stored as uint8 payloads.
    cl->n_virt = (cl->m_codes > 0)
        ? static_cast<int>(cl->virt_codes_small.size()) / cl->m_codes
        : 0;
    cl->nc = cl->n_real + cl->n_virt;
    cl->n_root_real = (cl->depth_offsets.size() >= 2)
        ? static_cast<int>(cl->depth_offsets[1])
        : CountRootRealFromLOUDS(*cl);

    if (use_coeff_codec) {
        if (!coeff_thr) {
            if (err) *err = "LOUDS-native eval: coeff reader missing.";
            return false;
        }
        Timer t_io;
        if (!coeff_thr->ReadCluster(cid,
                                    &cl->scales_root,
                                    &cl->scales_linkage,
                                    &cl->lens_root,
                                    &cl->lens_linkage,
                                    &cl->payload_root,
                                    &cl->payload_linkage,
                                    &local_err)) {
            if (err) *err = local_err;
            return false;
        }
        if (stats) stats->coeff_io_sec = t_io.ElapsedSeconds();

        const int streams = bundle->coeff_reader.meta().streams;
        const int S = bundle->coeff_reader.meta().S;
        const int base = (S == 256) ? 128 : (S / 2);
        cl->q_layer_major.assign(static_cast<std::size_t>(cl->m) * static_cast<std::size_t>(cl->nc), std::int8_t(0));
        Timer t_decode;
        DecodeGroupIntoRanges(cl->m, cl->nc, streams, S, base,
                              cl->lens_root, cl->payload_root,
                              0, cl->n_virt,
                              cl->n_virt, cl->n_virt + cl->n_root_real,
                              cl->q_layer_major.data());
        DecodeGroupIntoRanges(cl->m, cl->nc, streams, S, base,
                              cl->lens_linkage, cl->payload_linkage,
                              cl->n_virt + cl->n_root_real, cl->nc,
                              0, 0,
                              cl->q_layer_major.data());
        if (stats) stats->coeff_decode_sec = t_decode.ElapsedSeconds();
    }

    if (cl->n_real <= 0) {
        cl->r_norm2.clear();
        return true;
    }

    if (eval::IsDiskNorm2ModeLut(cfg.eval.disk_norm2_mode)) {
        if (bundle->norm2_lut_disk_present) {
            if (bundle->norm2_lut_global) {
                if (bundle->norm2_lut_disk_lazy_enabled) {
                    if (!eval::LoadNorm2LutClusterCodesSlice(*bundle->linkage_list,
                                                             use_coeff_codec,
                                                             cid,
                                                             &cl->norm2_lut.codes_u8,
                                                             err)) {
                        return false;
                    }
                } else {
                    const auto lut_it = bundle->norm2_lut_cache.find(cid);
                    if (lut_it == bundle->norm2_lut_cache.end()) {
                        if (err) {
                            *err = "LOUDS-native eval: global norm2 LUT cache exists on disk but cluster entry is missing (cid=" +
                                   std::to_string(cid) + "). Delete norm2_*_lut_* to force rebuild.";
                        }
                        return false;
                    }
                    cl->norm2_lut.codes_u8 = lut_it->second.codes_u8;
                }
            } else {
                if (bundle->norm2_lut_disk_lazy_enabled) {
                    if (!eval::LoadNorm2LutClusterSlice(*bundle->linkage_list,
                                                        use_coeff_codec,
                                                        cid,
                                                        &cl->norm2_lut,
                                                        err)) {
                        return false;
                    }
                } else {
                    const auto lut_it = bundle->norm2_lut_cache.find(cid);
                    if (lut_it == bundle->norm2_lut_cache.end()) {
                        if (err) {
                            *err = "LOUDS-native eval: norm2 LUT cache exists on disk but cluster entry is missing (cid=" +
                                   std::to_string(cid) + "). Delete norm2_*_lut_* to force rebuild.";
                        }
                        return false;
                    }
                    cl->norm2_lut = lut_it->second;
                }
            }
        } else {
            if (bundle->norm2_lut_global) {
                if (err) *err = "LOUDS-native eval: global norm2 LUT cache must be built before loading clusters.";
                return false;
            }
            if (!ComputeNorm2LutNative(use_coeff_codec, cfg, *bundle, cl, err)) {
                return false;
            }
        }
    } else {
        if (bundle->norm2_disk_present) {
            if (bundle->norm2_disk_lazy_enabled) {
                if (!eval::LoadFloatNorm2ClusterSlice(*bundle->linkage_list,
                                                      use_coeff_codec,
                                                      cid,
                                                      &cl->r_norm2,
                                                      err)) {
                    return false;
                }
            } else {
                const auto norm_it = bundle->norm2_cache.find(cid);
                if (norm_it == bundle->norm2_cache.end() ||
                    norm_it->second.size() != static_cast<std::size_t>(cl->n_real)) {
                    if (err) {
                        *err = "LOUDS-native eval: norm2 cache exists on disk but cluster entry is missing or mismatched (cid=" +
                               std::to_string(cid) + "). Delete norm2_*.f32 to force rebuild.";
                    }
                    return false;
                }
                cl->r_norm2 = norm_it->second;
            }
        } else {
            if (!ComputeNorm2Louds(use_coeff_codec, *bundle, cl, err)) return false;
        }
    }
    return true;
}

bool LoadCluster(bool use_coeff_codec,LoudsNativeBundle* bundle,
                 int cid,
                 const Config& cfg,
                 const TrainResult& train,
                 const LoudsNativeCluster** out,
                 std::string* err) {
    auto it = bundle->cluster_cache.find(cid);
    if (it != bundle->cluster_cache.end()) {
        if (out) *out = &it->second;
        return true;
    }

    LoudsNativeCluster cl;
    ClusterLoadStats stats{};
    if (!LoadClusterWithReaders(use_coeff_codec, bundle,
                                               bundle->linkage_thr,
                                               use_coeff_codec ? &bundle->coeff_thr : nullptr,
                                               cid,
                                               cfg,
                                               &cl,
                                               &stats,
                                               err)) {
        return false;
    }
    bundle->lazy_louds_wall_sec += (stats.louds_read_sec + stats.louds_deser_sec);
    bundle->lazy_huffman_wall_sec += (stats.coeff_io_sec + stats.coeff_decode_sec);
    bundle->lazy_louds_cpu_sec += stats.louds_deser_sec;
    bundle->lazy_huffman_cpu_sec += stats.coeff_decode_sec;

    auto [ins_it, _] = bundle->cluster_cache.emplace(cid, std::move(cl));
    if (out) *out = &ins_it->second;
    (void)train;
    return true;
}

bool PreloadAllClusters(bool use_coeff_codec,LoudsNativeBundle* bundle,
                        const Config& cfg,
                        const TrainResult& train,
                        std::string* err) {
    (void)train;
    if (bundle->preload_done) return true;
    const int preload_threads = cfg.eval.linkage_preload_clusters_io_threads;
    if (preload_threads < 0) return true;
    const int nlist = bundle->linkage_list->nlist();
    const int omp_default = GetOmpDefaultThreads();
    const int n_threads = std::max(1, std::min(
        (preload_threads == 0) ? std::max(1, omp_default) : preload_threads, nlist));
    bundle->preload_io_threads_used = n_threads;

    Timer preload_wall_timer;

    std::vector<io::LinkageListThreadReader> thr_list(static_cast<std::size_t>(n_threads));
    for (int t = 0; t < n_threads; ++t) {
        std::string e;
        if (!thr_list[static_cast<std::size_t>(t)].OpenFrom(*bundle->linkage_list, /*require_coeffs_f32=*/!use_coeff_codec, &e)) {
            if (err) *err = "LOUDS-native eval preload: list reader open failed: " + e;
            return false;
        }
    }
    std::vector<io::LinkageCoeffCodecThreadReader> thr_coeff(
        use_coeff_codec ? static_cast<std::size_t>(n_threads) : 0u);
    if (use_coeff_codec) {
        for (int t = 0; t < n_threads; ++t) {
            std::string e;
            if (!thr_coeff[static_cast<std::size_t>(t)].OpenFrom(bundle->coeff_reader, &e)) {
                if (err) *err = "LOUDS-native eval preload: coeff reader open failed: " + e;
                return false;
            }
        }
    }

    struct ClusterResult { int cid; LoudsNativeCluster cl; };
    const auto per_thread = static_cast<std::size_t>((nlist + n_threads - 1) / n_threads);
    std::vector<std::vector<ClusterResult>> results(static_cast<std::size_t>(n_threads));
    for (auto& v : results) v.reserve(per_thread);
    std::vector<std::string> thread_errs(static_cast<std::size_t>(n_threads));
    std::vector<bool> thread_ok(static_cast<std::size_t>(n_threads), true);
    std::vector<ClusterLoadStats> thread_stats(static_cast<std::size_t>(n_threads));

    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(n_threads));
    for (int t = 0; t < n_threads; ++t) {
        threads.emplace_back([&, t]() {
            const std::size_t ti = static_cast<std::size_t>(t);
            try {
                ClusterLoadStats sum{};
                for (int cid = t; cid < nlist; cid += n_threads) {
                    LoudsNativeCluster cl;
                    std::string local_err;
                    ClusterLoadStats st{};
                    if (!LoadClusterWithReaders(use_coeff_codec, bundle,
                                                               thr_list[ti],
                                                               use_coeff_codec ? &thr_coeff[ti] : nullptr,
                                                               cid,
                                                               cfg,
                                                               &cl,
                                                               &st,
                                                               &local_err)) {
                        thread_ok[ti] = false;
                        thread_errs[ti] = local_err;
                        return;
                    }
                    sum.louds_read_sec += st.louds_read_sec;
                    sum.louds_deser_sec += st.louds_deser_sec;
                    sum.coeff_io_sec += st.coeff_io_sec;
                    sum.coeff_decode_sec += st.coeff_decode_sec;
                    results[ti].push_back(ClusterResult{cid, std::move(cl)});
                }
                thread_stats[ti] = sum;
            } catch (const std::exception& e) {
                thread_ok[ti] = false;
                thread_errs[ti] = e.what();
                return;
            } catch (...) {
                thread_ok[ti] = false;
                thread_errs[ti] = "unknown exception";
                return;
            }
        });
    }
    for (auto& th : threads) th.join();
    for (int t = 0; t < n_threads; ++t) {
        if (!thread_ok[static_cast<std::size_t>(t)]) {
            if (err) *err = "LOUDS-native eval preload: thread " + std::to_string(t) +
                             " failed: " + thread_errs[static_cast<std::size_t>(t)];
            return false;
        }
    }

    for (auto& vec : results) {
        for (auto& r : vec) {
            bundle->cluster_cache.emplace(r.cid, std::move(r.cl));
        }
    }

    // Preload component timing: approximate critical-path wall via max per-thread sums.
    // This avoids "sum over clusters" reporting while still providing a wall-like estimate.
    double louds_wall_cp = 0.0;
    double louds_cpu_cp = 0.0;
    double huff_wall_cp = 0.0;
    double huff_cpu_cp = 0.0;
    for (int t = 0; t < n_threads; ++t) {
        const ClusterLoadStats& s = thread_stats[static_cast<std::size_t>(t)];
        louds_wall_cp = std::max(louds_wall_cp, s.louds_read_sec + s.louds_deser_sec);
        louds_cpu_cp = std::max(louds_cpu_cp, s.louds_deser_sec);
        huff_wall_cp = std::max(huff_wall_cp, s.coeff_io_sec + s.coeff_decode_sec);
        huff_cpu_cp = std::max(huff_cpu_cp, s.coeff_decode_sec);
    }
    bundle->preload_louds_wall_sec = louds_wall_cp;
    bundle->preload_louds_cpu_sec = louds_cpu_cp;
    bundle->preload_huffman_wall_sec = huff_wall_cp;
    bundle->preload_huffman_cpu_sec = huff_cpu_cp;

    bundle->preload_wall_sec = preload_wall_timer.ElapsedSeconds();
    bundle->preload_done = true;
    return true;
}

bool EnsureLoudsNativeBundle(bool use_coeff_codec,const Config& cfg,
                            const io::LinkageListReader& linkage_list,
                            const TrainResult& train,
                            DiskLinkageEvalTiming* /*timing*/,
                            DiskLinkageEvalSession* session,
                            std::shared_ptr<LoudsNativeBundle>* bundle_out,
                            std::string* err) {
    std::shared_ptr<LoudsNativeBundle> bundle;
    const bool first_call = !(session && session->provider_cache);
    if (!first_call) {
        bundle = std::static_pointer_cast<LoudsNativeBundle>(session->provider_cache);
    } else {
        bundle = std::make_shared<LoudsNativeBundle>();
        if (eval::IsDiskNorm2ModeLutGlobal(cfg.eval.disk_norm2_mode)) {
            std::string lut_err;
            const auto check_res = eval::CheckNorm2LutCache(linkage_list, use_coeff_codec,
                                                            cfg.base.encode.hnorms,
                                                            cfg.eval.disk_norm2_lut_kmeans_niter,
                                                            cfg.eval.disk_norm2_mode,
                                                            cfg.eval.disk_norm2_lut_log_alpha,
                                                            cfg.eval.disk_norm2_lut_piecewise_p1,
                                                            cfg.eval.disk_norm2_lut_piecewise_p2,
                                                            cfg.eval.disk_norm2_lut_piecewise_count_weight,
                                                            cfg.eval.disk_norm2_lut_piecewise_range_weight,
                                                            &lut_err);
            if (check_res != eval::Norm2LutDiskLoadResult::kLoaded) {
                if (check_res == eval::Norm2LutDiskLoadResult::kInvalid) {
                    LogWarn("LOUDS-native eval: rebuilding stale global norm2 LUT cache (" +
                            (lut_err.empty() ? std::string("invalid cache") : lut_err) + ").");
                } else {
                    LogInfo("Precomputing full global norm2 LUT cache ...");
                }
                if (!BuildAndStoreFullNorm2LutCacheNative(use_coeff_codec, cfg, linkage_list, train, err)) {
                    return false;
                }
            }
        }
        if (!OpenBundle(use_coeff_codec, cfg, linkage_list, train, bundle.get(), err)) return false;
        if (session) session->provider_cache = bundle;
    }

    if (!bundle->setup_logged && !cfg.eval.bench_quiet) {
        LogInfo("Linkage recall: using LOUDS-native CPU eval path (direct ParentLOUDS queries; no parent[] materialization).");
        if (cfg.eval.linkage_gpu_scan_enable || cfg.runtime.use_cuda) {
            LogInfo("Linkage recall: LOUDS-native path is CPU-only in this implementation; GPU eval toggles are ignored.");
        }
        bundle->setup_logged = true;
    }

    const bool is_preload_mode = (cfg.eval.linkage_preload_clusters_io_threads >= 0);
    if (is_preload_mode && !bundle->preload_done && !cfg.eval.bench_quiet) {
        const int io_thr = cfg.eval.linkage_preload_clusters_io_threads;
        const int omp_default = GetOmpDefaultThreads();
        const int actual_threads = std::max(1, std::min(
            (io_thr == 0) ? std::max(1, omp_default) : io_thr, linkage_list.nlist()));
        LogInfo("Linkage recall: preloading all cluster data (LOUDS-native CPU path, io_threads=" +
                std::string(io_thr == 0 ? "follow_omp" : std::to_string(io_thr)) +
                ", actual_io_threads=" + std::to_string(actual_threads) + ") ...");
        if (!PreloadAllClusters(use_coeff_codec, bundle.get(), cfg, train, err)) return false;
    } else {
        if (!PreloadAllClusters(use_coeff_codec, bundle.get(), cfg, train, err)) return false;
    }

    if (!bundle->load_mode_logged && !cfg.eval.bench_quiet) {
        const std::string omp_decode_note = "omp_decode_threads=" + std::to_string(OmpMaxThreads());
        if (is_preload_mode) {
            LogInfo("LOUDS/Huffman load mode = preload-native (preload_io_threads=" +
                    std::to_string(bundle->preload_io_threads_used) + ", " + omp_decode_note + ")");

            // Public bench/prep timing output: report once here (before per-repeat eval logs).
            if (!bundle->timing_reported_louds) {
                LogInfo("Disk IVF linkage parent LOUDS native prep louds wall/cpu: " +
                        std::to_string(bundle->preload_louds_wall_sec) + "s / " +
                        std::to_string(bundle->preload_louds_cpu_sec) + "s");
                bundle->timing_reported_louds = true;
            }
            if (use_coeff_codec && !bundle->timing_reported_huffman) {
                LogInfo("Disk IVF linkage parent LOUDS native prep huffman wall/cpu: " +
                        std::to_string(bundle->preload_huffman_wall_sec) + "s / " +
                        std::to_string(bundle->preload_huffman_cpu_sec) + "s");
                bundle->timing_reported_huffman = true;
            }
        } else {
            LogInfo("LOUDS/Huffman load mode = lazy-native (" + omp_decode_note + ")");
        }
        bundle->load_mode_logged = true;
    }

    if (bundle_out) *bundle_out = bundle;
    return true;
}

void ScanOneQueryFloat(const LoudsNativeCluster& cl,
                       const float* xCq_root0_col,
                       const int* offsets_root_small,
                       const float* xCq_root_small_col,
                       const int* offsets_one,
                       const float* xCq_one_col,
                       TopKHeap* topk,
                       ScanScratch* scratch) {
    const int m = cl.m;
    const int m_codes = cl.m_codes;
    const int n_real = cl.n_real;
    const int n_virt = cl.n_virt;
    const int nc = cl.nc;
    const int real_base = n_virt;

    scratch->dot_all.resize(static_cast<std::size_t>(nc));
    float* dot_all = scratch->dot_all.data();
    const float root0 = xCq_root0_col[cl.cid];
    const std::ptrdiff_t stride_codes = static_cast<std::ptrdiff_t>(m_codes);

    const float* rootptr[64] = {};
    const float* oneptr[64] = {};
    oneptr[0] = xCq_one_col + offsets_one[0];
    for (int l = 1; l < m; ++l) {
        rootptr[l] = xCq_root_small_col + offsets_root_small[l];
        oneptr[l] = xCq_one_col + offsets_one[l];
    }

    #pragma omp simd
    for (int v = 0; v < n_virt; ++v) dot_all[v] = cl.virt_a0[static_cast<std::size_t>(v)] * root0;
    #pragma omp simd
    for (int pos = 0; pos < cl.n_root_real; ++pos) dot_all[real_base + pos] = cl.a0[static_cast<std::size_t>(pos)] * root0;

#if defined(__GNUC__)
    #pragma GCC unroll 16
#elif defined(_MSC_VER)
    #pragma loop( unroll(16) )
#endif
    for (int l = 1; l < m; ++l) {
        const float* table = rootptr[l];
        for (int v = 0; v < n_virt; ++v) {
            const std::uint8_t* row = cl.virt_codes_small.data() + static_cast<std::ptrdiff_t>(v) * stride_codes;
            const float* arow = cl.virt_coeffs_small.data() + static_cast<std::ptrdiff_t>(v) * m_codes;
                dot_all[v] += arow[l - 1] * table[static_cast<int>(row[l - 1])];
        }
        for (int pos = 0; pos < cl.n_root_real; ++pos) {
            const std::uint8_t* row = cl.codes_small.data() + static_cast<std::ptrdiff_t>(pos) * stride_codes;
            const float* arow = cl.coeffs_small.data() + static_cast<std::ptrdiff_t>(pos) * m_codes;
            const int local = real_base + pos;
                dot_all[local] += arow[l - 1] * table[static_cast<int>(row[l - 1])];
        }
    }

    const float* one0 = oneptr[0];
    auto parent_decoder = cl.louds.MakeSequentialParentDecoder();
    parent_decoder.Skip(static_cast<std::size_t>(real_base + cl.n_root_real));
    for (int pos = cl.n_root_real; pos < n_real; ++pos) {
        const int local = real_base + pos;
        const int p = static_cast<int>(parent_decoder.NextParent1Based()) - 1;
        float dot = 0.0f;
        const int code0 = static_cast<int>(cl.code0_one[static_cast<std::size_t>(pos)]);
        dot += cl.a0[static_cast<std::size_t>(pos)] * one0[code0];
        const std::uint8_t* row = cl.codes_small.data() + static_cast<std::ptrdiff_t>(pos) * stride_codes;
        const float* arow = cl.coeffs_small.data() + static_cast<std::ptrdiff_t>(pos) * m_codes;
        for (int l = 1; l < m; ++l) {
            const float* one_l = oneptr[l];
            dot += arow[l - 1] * one_l[static_cast<int>(row[l - 1])];
        }
        dot_all[local] = dot + dot_all[p];
    }

    scratch->dist_tmp.resize(static_cast<std::size_t>(n_real));
    float* dtmp = scratch->dist_tmp.data();
    #pragma omp simd
    for (int pos = 0; pos < n_real; ++pos) dtmp[pos] = cl.r_norm2[static_cast<std::size_t>(pos)] - 2.0f * dot_all[real_base + pos];

    const int heap_k = topk->k;
    if (static_cast<int>(topk->heap.size()) < heap_k) {
        for (int pos = 0; pos < n_real; ++pos) topk->Push(dtmp[pos], static_cast<std::int32_t>(cl.real_ids[static_cast<std::size_t>(pos)]));
    } else {
        if (!topk->heapified) topk->Heapify();
        Candidate worst = topk->heap[0];
        for (int pos = 0; pos < n_real; ++pos) {
            const float dist = dtmp[pos];
            const auto id = static_cast<std::int32_t>(cl.real_ids[static_cast<std::size_t>(pos)]);
            if (dist > worst.dist || (dist == worst.dist && id >= worst.id)) continue;
            topk->PushAssumeBetter(dist, id);
            worst = topk->heap[0];
        }
    }
}

void ScanOneQueryFloatLut(const LoudsNativeCluster& cl,
                          const std::uint8_t* norm2_codes,
                          const float* norm2_centers,
                          const float* xCq_root0_col,
                          const int* offsets_root_small,
                          const float* xCq_root_small_col,
                          const int* offsets_one,
                          const float* xCq_one_col,
                          TopKHeap* topk,
                          ScanScratch* scratch) {
    const int m = cl.m;
    const int m_codes = cl.m_codes;
    const int n_real = cl.n_real;
    const int n_virt = cl.n_virt;
    const int nc = cl.nc;
    const int real_base = n_virt;

    scratch->dot_all.resize(static_cast<std::size_t>(nc));
    float* dot_all = scratch->dot_all.data();
    const float root0 = xCq_root0_col[cl.cid];
    const std::ptrdiff_t stride_codes = static_cast<std::ptrdiff_t>(m_codes);

    const float* rootptr[64] = {};
    const float* oneptr[64] = {};
    oneptr[0] = xCq_one_col + offsets_one[0];
    for (int l = 1; l < m; ++l) {
        rootptr[l] = xCq_root_small_col + offsets_root_small[l];
        oneptr[l] = xCq_one_col + offsets_one[l];
    }

    #pragma omp simd
    for (int v = 0; v < n_virt; ++v) dot_all[v] = cl.virt_a0[static_cast<std::size_t>(v)] * root0;
    #pragma omp simd
    for (int pos = 0; pos < cl.n_root_real; ++pos) dot_all[real_base + pos] = cl.a0[static_cast<std::size_t>(pos)] * root0;

#if defined(__GNUC__)
    #pragma GCC unroll 16
#elif defined(_MSC_VER)
    #pragma loop( unroll(16) )
#endif
    for (int l = 1; l < m; ++l) {
        const float* table = rootptr[l];
        for (int v = 0; v < n_virt; ++v) {
            const std::uint8_t* row = cl.virt_codes_small.data() + static_cast<std::ptrdiff_t>(v) * stride_codes;
            const float* arow = cl.virt_coeffs_small.data() + static_cast<std::ptrdiff_t>(v) * m_codes;
            dot_all[v] += arow[l - 1] * table[static_cast<int>(row[l - 1])];
        }
        for (int pos = 0; pos < cl.n_root_real; ++pos) {
            const std::uint8_t* row = cl.codes_small.data() + static_cast<std::ptrdiff_t>(pos) * stride_codes;
            const float* arow = cl.coeffs_small.data() + static_cast<std::ptrdiff_t>(pos) * m_codes;
            const int local = real_base + pos;
            dot_all[local] += arow[l - 1] * table[static_cast<int>(row[l - 1])];
        }
    }

    const float* one0 = oneptr[0];
    auto parent_decoder = cl.louds.MakeSequentialParentDecoder();
    parent_decoder.Skip(static_cast<std::size_t>(real_base + cl.n_root_real));
    for (int pos = cl.n_root_real; pos < n_real; ++pos) {
        const int local = real_base + pos;
        const int p = static_cast<int>(parent_decoder.NextParent1Based()) - 1;
        float dot = 0.0f;
        const int code0 = static_cast<int>(cl.code0_one[static_cast<std::size_t>(pos)]);
        dot += cl.a0[static_cast<std::size_t>(pos)] * one0[code0];
        const std::uint8_t* row = cl.codes_small.data() + static_cast<std::ptrdiff_t>(pos) * stride_codes;
        const float* arow = cl.coeffs_small.data() + static_cast<std::ptrdiff_t>(pos) * m_codes;
        for (int l = 1; l < m; ++l) {
            const float* one_l = oneptr[l];
            dot += arow[l - 1] * one_l[static_cast<int>(row[l - 1])];
        }
        dot_all[local] = dot + dot_all[p];
    }

    scratch->dist_tmp.resize(static_cast<std::size_t>(n_real));
    float* dtmp = scratch->dist_tmp.data();
    #pragma omp simd
    for (int pos = 0; pos < n_real; ++pos) {
        dtmp[pos] = norm2_centers[static_cast<int>(norm2_codes[static_cast<std::size_t>(pos)])] -
                    2.0f * dot_all[real_base + pos];
    }

    const int heap_k = topk->k;
    if (static_cast<int>(topk->heap.size()) < heap_k) {
        for (int pos = 0; pos < n_real; ++pos) topk->Push(dtmp[pos], static_cast<std::int32_t>(cl.real_ids[static_cast<std::size_t>(pos)]));
    } else {
        if (!topk->heapified) topk->Heapify();
        Candidate worst = topk->heap[0];
        for (int pos = 0; pos < n_real; ++pos) {
            const float dist = dtmp[pos];
            const auto id = static_cast<std::int32_t>(cl.real_ids[static_cast<std::size_t>(pos)]);
            if (dist > worst.dist || (dist == worst.dist && id >= worst.id)) continue;
            topk->PushAssumeBetter(dist, id);
            worst = topk->heap[0];
        }
    }
}

void ScanOneQueryInt8(const LoudsNativeCluster& cl,
                      const float* xCq_root0_col,
                      const int* offsets_root_small,
                      const float* xCq_root_small_col,
                      int root_small_total_cols,
                      const int* offsets_one,
                      const float* xCq_one_col,
                      int one_total_cols,
                      TopKHeap* topk,
                      ScanScratch* scratch) {
    const int m = cl.m;
    const int m_codes = cl.m_codes;
    const int n_real = cl.n_real;
    const int n_virt = cl.n_virt;
    const int nc = cl.nc;
    const int real_base = n_virt;

    scratch->dot_all.resize(static_cast<std::size_t>(nc));
    scratch->root_small_scaled.resize(static_cast<std::size_t>(root_small_total_cols));
    scratch->one_scaled.resize(static_cast<std::size_t>(one_total_cols));
    float* dot_all = scratch->dot_all.data();
    CopyScalePackedTable(scratch->root_small_scaled.data(), xCq_root_small_col, offsets_root_small,
                         root_small_total_cols, cl.scales_root.data(), m, 1);
    CopyScalePackedTable(scratch->one_scaled.data(), xCq_one_col, offsets_one,
                         one_total_cols, cl.scales_linkage.data(), m, 0);
    const float root0_scaled = cl.scales_root[0] * xCq_root0_col[cl.cid];
    const float* root_small = scratch->root_small_scaled.data();
    const float* one = scratch->one_scaled.data();
    const std::ptrdiff_t stride_codes = static_cast<std::ptrdiff_t>(m_codes);

    const std::int8_t* qptr[64] = {};
    const float* oneptr[64] = {};
    for (int l = 0; l < m; ++l) {
        qptr[l] = cl.q_layer_major.data() + static_cast<std::ptrdiff_t>(l) * nc;
        oneptr[l] = one + offsets_one[l];
    }
    const std::int8_t* q0 = qptr[0];

    #pragma omp simd
    for (int v = 0; v < n_virt; ++v) dot_all[v] = root0_scaled * static_cast<float>(q0[v]);
    #pragma omp simd
    for (int pos = 0; pos < cl.n_root_real; ++pos) {
        const int local = real_base + pos;
        dot_all[local] = root0_scaled * static_cast<float>(q0[local]);
    }
#if defined(__GNUC__)
    #pragma GCC unroll 16
#elif defined(_MSC_VER)
    #pragma loop( unroll(16) )
#endif
    for (int l = 1; l < m; ++l) {
        const std::int8_t* ql = qptr[l];
        const float* table = root_small + offsets_root_small[l];
        for (int v = 0; v < n_virt; ++v) {
            const std::uint8_t* row = cl.virt_codes_small.data() + static_cast<std::ptrdiff_t>(v) * stride_codes;
                dot_all[v] += static_cast<float>(ql[v]) * table[static_cast<int>(row[l - 1])];
        }
        for (int pos = 0; pos < cl.n_root_real; ++pos) {
            const int local = real_base + pos;
            const std::uint8_t* row = cl.codes_small.data() + static_cast<std::ptrdiff_t>(pos) * stride_codes;
                dot_all[local] += static_cast<float>(ql[local]) * table[static_cast<int>(row[l - 1])];
        }
    }

    const float* one0 = oneptr[0];
    auto parent_decoder = cl.louds.MakeSequentialParentDecoder();
    parent_decoder.Skip(static_cast<std::size_t>(real_base + cl.n_root_real));
    for (int pos = cl.n_root_real; pos < n_real; ++pos) {
        const int local = real_base + pos;
        const int p = static_cast<int>(parent_decoder.NextParent1Based()) - 1;
        float dot = 0.0f;
        const int code0 = static_cast<int>(cl.code0_one[static_cast<std::size_t>(pos)]);
        dot += static_cast<float>(q0[local]) * one0[code0];
        const std::uint8_t* row = cl.codes_small.data() + static_cast<std::ptrdiff_t>(pos) * stride_codes;
        for (int l = 1; l < m; ++l) {
            const std::int8_t* ql = qptr[l];
            const float* one_l = oneptr[l];
                dot += static_cast<float>(ql[local]) * one_l[static_cast<int>(row[l - 1])];
        }
        dot_all[local] = dot + dot_all[p];
    }

    scratch->dist_tmp.resize(static_cast<std::size_t>(n_real));
    float* dtmp = scratch->dist_tmp.data();
    #pragma omp simd
    for (int pos = 0; pos < n_real; ++pos) dtmp[pos] = cl.r_norm2[static_cast<std::size_t>(pos)] - 2.0f * dot_all[real_base + pos];

    const int heap_k = topk->k;
    if (static_cast<int>(topk->heap.size()) < heap_k) {
        for (int pos = 0; pos < n_real; ++pos) topk->Push(dtmp[pos], static_cast<std::int32_t>(cl.real_ids[static_cast<std::size_t>(pos)]));
    } else {
        if (!topk->heapified) topk->Heapify();
        Candidate worst = topk->heap[0];
        for (int pos = 0; pos < n_real; ++pos) {
            const float dist = dtmp[pos];
            const auto id = static_cast<std::int32_t>(cl.real_ids[static_cast<std::size_t>(pos)]);
            if (dist > worst.dist || (dist == worst.dist && id >= worst.id)) continue;
            topk->PushAssumeBetter(dist, id);
            worst = topk->heap[0];
        }
    }
}

void ScanOneQueryInt8Lut(const LoudsNativeCluster& cl,
                         const std::uint8_t* norm2_codes,
                         const float* norm2_centers,
                         const float* xCq_root0_col,
                         const int* offsets_root_small,
                         const float* xCq_root_small_col,
                         int root_small_total_cols,
                         const int* offsets_one,
                         const float* xCq_one_col,
                         int one_total_cols,
                         TopKHeap* topk,
                         ScanScratch* scratch) {
    const int m = cl.m;
    const int m_codes = cl.m_codes;
    const int n_real = cl.n_real;
    const int n_virt = cl.n_virt;
    const int nc = cl.nc;
    const int real_base = n_virt;

    scratch->dot_all.resize(static_cast<std::size_t>(nc));
    scratch->root_small_scaled.resize(static_cast<std::size_t>(root_small_total_cols));
    scratch->one_scaled.resize(static_cast<std::size_t>(one_total_cols));
    float* dot_all = scratch->dot_all.data();
    CopyScalePackedTable(scratch->root_small_scaled.data(), xCq_root_small_col, offsets_root_small,
                         root_small_total_cols, cl.scales_root.data(), m, 1);
    CopyScalePackedTable(scratch->one_scaled.data(), xCq_one_col, offsets_one,
                         one_total_cols, cl.scales_linkage.data(), m, 0);
    const float root0_scaled = cl.scales_root[0] * xCq_root0_col[cl.cid];
    const float* root_small = scratch->root_small_scaled.data();
    const float* one = scratch->one_scaled.data();
    const std::ptrdiff_t stride_codes = static_cast<std::ptrdiff_t>(m_codes);

    const std::int8_t* qptr[64] = {};
    const float* oneptr[64] = {};
    for (int l = 0; l < m; ++l) {
        qptr[l] = cl.q_layer_major.data() + static_cast<std::ptrdiff_t>(l) * nc;
        oneptr[l] = one + offsets_one[l];
    }
    const std::int8_t* q0 = qptr[0];

    #pragma omp simd
    for (int v = 0; v < n_virt; ++v) dot_all[v] = root0_scaled * static_cast<float>(q0[v]);
    #pragma omp simd
    for (int pos = 0; pos < cl.n_root_real; ++pos) {
        const int local = real_base + pos;
        dot_all[local] = root0_scaled * static_cast<float>(q0[local]);
    }
#if defined(__GNUC__)
    #pragma GCC unroll 16
#elif defined(_MSC_VER)
    #pragma loop( unroll(16) )
#endif
    for (int l = 1; l < m; ++l) {
        const std::int8_t* ql = qptr[l];
        const float* table = root_small + offsets_root_small[l];
        for (int v = 0; v < n_virt; ++v) {
            const std::uint8_t* row = cl.virt_codes_small.data() + static_cast<std::ptrdiff_t>(v) * stride_codes;
            dot_all[v] += static_cast<float>(ql[v]) * table[static_cast<int>(row[l - 1])];
        }
        for (int pos = 0; pos < cl.n_root_real; ++pos) {
            const int local = real_base + pos;
            const std::uint8_t* row = cl.codes_small.data() + static_cast<std::ptrdiff_t>(pos) * stride_codes;
            dot_all[local] += static_cast<float>(ql[local]) * table[static_cast<int>(row[l - 1])];
        }
    }

    const float* one0 = oneptr[0];
    auto parent_decoder = cl.louds.MakeSequentialParentDecoder();
    parent_decoder.Skip(static_cast<std::size_t>(real_base + cl.n_root_real));
    for (int pos = cl.n_root_real; pos < n_real; ++pos) {
        const int local = real_base + pos;
        const int p = static_cast<int>(parent_decoder.NextParent1Based()) - 1;
        float dot = 0.0f;
        const int code0 = static_cast<int>(cl.code0_one[static_cast<std::size_t>(pos)]);
        dot += static_cast<float>(q0[local]) * one0[code0];
        const std::uint8_t* row = cl.codes_small.data() + static_cast<std::ptrdiff_t>(pos) * stride_codes;
        for (int l = 1; l < m; ++l) {
            const std::int8_t* ql = qptr[l];
            const float* one_l = oneptr[l];
            dot += static_cast<float>(ql[local]) * one_l[static_cast<int>(row[l - 1])];
        }
        dot_all[local] = dot + dot_all[p];
    }

    scratch->dist_tmp.resize(static_cast<std::size_t>(n_real));
    float* dtmp = scratch->dist_tmp.data();
    #pragma omp simd
    for (int pos = 0; pos < n_real; ++pos) {
        dtmp[pos] = norm2_centers[static_cast<int>(norm2_codes[static_cast<std::size_t>(pos)])] -
                    2.0f * dot_all[real_base + pos];
    }

    const int heap_k = topk->k;
    if (static_cast<int>(topk->heap.size()) < heap_k) {
        for (int pos = 0; pos < n_real; ++pos) topk->Push(dtmp[pos], static_cast<std::int32_t>(cl.real_ids[static_cast<std::size_t>(pos)]));
    } else {
        if (!topk->heapified) topk->Heapify();
        Candidate worst = topk->heap[0];
        for (int pos = 0; pos < n_real; ++pos) {
            const float dist = dtmp[pos];
            const auto id = static_cast<std::int32_t>(cl.real_ids[static_cast<std::size_t>(pos)]);
            if (dist > worst.dist || (dist == worst.dist && id >= worst.id)) continue;
            topk->PushAssumeBetter(dist, id);
            worst = topk->heap[0];
        }
    }
}

bool EvalImpl(const Config& cfg, bool use_coeff_codec,
              const Dataset& query_dataset_inmem,
              const io::LinkageListReader& linkage_list,
              const TrainResult& train,
              RecallResult* out,
              DiskLinkageEvalTiming* timing,
              DiskLinkageEvalSession* session,
              std::string* err) {
    // qt_rotate_wall_sec is an INPUT from the caller.
    // Query rotation is performed before entering the timed repeat loop in main_virtual,
    // so the same measured rotate cost is intentionally reused across repeats.
    const double qt_rotate_wall_sec = timing ? timing->qt_rotate_wall_sec : 0.0;
    if (!cfg.eval.linkage_use_ivf_disk) {
        if (err) *err = "LOUDS-native eval: eval.linkage.use_ivf_disk is false.";
        return false;
    }
    if (!cfg.eval.linkage_parent_louds_enable || !cfg.eval.linkage_parent_louds_native_eval) {
        if (err) *err = "LOUDS-native eval: mode not enabled.";
        return false;
    }
    if (!linkage_list.has_parent_louds()) {
        if (err) *err = "LOUDS-native eval: linkage_list store has no parent LOUDS blobs.";
        return false;
    }
    if (!out) return false;
    const int nquery = query_dataset_inmem.Xq.cols;
    const int d = query_dataset_inmem.Xq.rows;
    const int m = cfg.model.m;
    const int m_codes = std::max(0, m - 1);
    if (query_dataset_inmem.gt.size() != static_cast<std::size_t>(nquery)) {
        if (err) *err = "LOUDS-native eval: invalid ground truth.";
        return false;
    }
    if (d <= 0 || m <= 1 || m_codes != linkage_list.m_codes()) {
        if (err) *err = "LOUDS-native eval: invalid dims / m_codes mismatch.";
        return false;
    }
    const int k = std::max(1, std::min(cfg.dataset.k, static_cast<int>(linkage_list.total_real())));
    const int nprobe_cap = std::min(std::max(1, cfg.eval.linkage_nprobe), linkage_list.nlist());
    const int qblk = std::max(1, cfg.eval.linkage_query_block);
    const bool use_norm2_lut = eval::IsDiskNorm2ModeLut(cfg.eval.disk_norm2_mode);
    out->indices = ColMajorMatrix<int>(k, nquery);
    out->dists = ColMajorMatrix<float>(k, nquery);

    std::shared_ptr<LoudsNativeBundle> bundle;
    if (!EnsureLoudsNativeBundle(use_coeff_codec, cfg, linkage_list, train, timing, session, &bundle, err)) return false;

    const ColMajorMatrix<float>& C_root0 = train.C_root.books.front();
    std::vector<const ColMajorMatrix<float>*> root_small_books;
    root_small_books.reserve(static_cast<std::size_t>(m_codes));
    for (int l = 1; l < m; ++l) root_small_books.push_back(&train.C_root.books[static_cast<std::size_t>(l)]);
    const CodebookMeta meta_root_small = BuildCodebookMeta(root_small_books);
    const CodebookMeta meta_one = BuildCodebookMeta(GatherBooks(train.C_one));

    std::vector<int> offsets_root_small(static_cast<std::size_t>(m), 0);
    for (int l = 1; l < m; ++l) offsets_root_small[static_cast<std::size_t>(l)] = meta_root_small.offsets[static_cast<std::size_t>(l - 1)];
    [[maybe_unused]] const int root_small_total_cols = meta_root_small.total_cols;
    [[maybe_unused]] const int one_total_cols = meta_one.total_cols;

    double total_core_qt_wall = 0.0;
    double total_core_qt_gemm_wall = 0.0;
    double total_core_qt_coarse_gemm_wall = 0.0;
    double total_core_qt_root_small_gemm_wall = 0.0;
    double total_core_qt_one_gemm_wall = 0.0;
    double total_core_probe_sel_wall = 0.0;
    double total_core_scan_topk_wall = 0.0;

    const std::string probe_mode = cfg.eval.linkage_ivf_probe_mode;
    if (probe_mode != "exact" && probe_mode != "hier2" && probe_mode != "hnsw") {
        if (err) *err = "LOUDS-native eval: unsupported eval.linkage.ivf_probe_mode: " + probe_mode + " (expected exact|hier2|hnsw)";
        return false;
    }
    const bool probe_exact = (probe_mode == "exact");
    const bool probe_hier2 = (probe_mode == "hier2");
    const bool probe_hnsw = (probe_mode == "hnsw");

    std::vector<float> root_inv_norm_exact;
    if (probe_exact) {
        root_inv_norm_exact.assign(static_cast<std::size_t>(linkage_list.nlist()), 1.0f);
        for (int cid = 0; cid < linkage_list.nlist(); ++cid) {
            const float* c = C_root0.Col(cid);
            double ss = 0.0;
            for (int r = 0; r < d; ++r) {
                const auto v = static_cast<double>(c[r]);
                ss += v * v;
            }
            root_inv_norm_exact[static_cast<std::size_t>(cid)] =
                1.0f / std::sqrt(std::max(static_cast<float>(ss), 1e-20f));
        }
    }
    ivf::Hier2Split hier2;
    int hier2_topL = 1;
    ColMajorMatrix<float> C_coarse;
    std::vector<float> root_inv_norm_hier2;
    ColMajorMatrix<float> C_root0_unit;
    if (probe_hier2) {
        const int nlist = linkage_list.nlist();
        const int K1 = ivf::ComputeHier2DefaultK1(nlist);
        const int kmeans_iters = std::max(1, cfg.eval.linkage_ivf_hier2_kmeans_niter);
        const unsigned seed = 12345u;
        const std::filesystem::path cache_dir =
            DefaultHier2SplitCacheDirFromLinkageListDir(std::filesystem::path(linkage_list.dir()));

        std::uint64_t unit_hash = 0x9e3779b97f4a7c15ULL;
        unit_hash = Mix64(unit_hash, 1ULL);  // version
        unit_hash = Mix64(unit_hash, static_cast<std::uint64_t>(d));
        unit_hash = Mix64(unit_hash, static_cast<std::uint64_t>(nlist));

        root_inv_norm_hier2.assign(static_cast<std::size_t>(linkage_list.nlist()), 1.0f);
        C_root0_unit.rows = d;
        C_root0_unit.cols = linkage_list.nlist();
        C_root0_unit.data.resize(C_root0.data.size(), 0.0f);
        for (int cid = 0; cid < linkage_list.nlist(); ++cid) {
            const float* c = C_root0.Col(cid);
            float* c_unit = C_root0_unit.Col(cid);
            double ss = 0.0;
            for (int r = 0; r < d; ++r) {
                const auto v = static_cast<double>(c[r]);
                ss += v * v;
            }
            const float inv_norm =
                1.0f / std::sqrt(std::max(static_cast<float>(ss), 1e-20f));
            root_inv_norm_hier2[static_cast<std::size_t>(cid)] = inv_norm;
            for (int r = 0; r < d; ++r) {
                const float v = c[r] * inv_norm;
                c_unit[r] = v;
                std::uint32_t u = 0;
                static_assert(sizeof(float) == sizeof(std::uint32_t));
                std::memcpy(&u, &v, sizeof(u));
                unit_hash = Mix64(unit_hash, static_cast<std::uint64_t>(u));
            }
        }

        bool loaded = false;
        {
            ivf::Hier2Split cached;
            if (!TryLoadHier2SplitCache(cache_dir, d, nlist, K1, kmeans_iters, seed, unit_hash, &cached)) {
                // treat as miss
            } else if (cached.K == nlist && cached.K1 == K1 && !cached.groups.empty()) {
                hier2 = std::move(cached);
                loaded = true;
            }
        }
        if (!loaded) {
            if (!bundle || !bundle->hier2_logged) {
                LogInfo("[hier2] building spherical k-means split (first time; will be cached) nlist=" +
                        std::to_string(nlist) + " K1=" + std::to_string(K1) +
                        " iters=" + std::to_string(kmeans_iters) +
                        " dir=" + cache_dir.string());
            }
            hier2 = ivf::BuildHier2SplitKmeans(C_root0_unit.data.data(), d, nlist, K1,
                                               kmeans_iters, seed);
            TryWriteHier2SplitCache(cache_dir, d, nlist, hier2, kmeans_iters, seed, unit_hash);
        } else {
            if (!bundle || !bundle->hier2_logged) {
                LogInfo("[hier2] loaded cached split nlist=" + std::to_string(nlist) +
                        " K1=" + std::to_string(K1) +
                        " iters=" + std::to_string(kmeans_iters) +
                        " dir=" + cache_dir.string());
            }
        }
        hier2_topL = std::max(1, std::min(hier2.K1, cfg.eval.linkage_ivf_hier2_top_coarse));
        std::vector<float> coarse_flat = ivf::BuildCoarseFromHier2(C_root0_unit.data.data(), d, hier2);
        C_coarse.rows = d;
        C_coarse.cols = hier2.K1;
        C_coarse.data = std::move(coarse_flat);
        if (!bundle || !bundle->hier2_logged) {
            int min_sz = linkage_list.nlist(), max_sz = 0;
            double sum_sz = 0.0;
            for (int c = 0; c < hier2.K1; ++c) {
                const int sz = static_cast<int>(hier2.groups[static_cast<std::size_t>(c)].size());
                min_sz = std::min(min_sz, sz);
                max_sz = std::max(max_sz, sz);
                sum_sz += sz;
            }
            LogInfo("[hier2] spherical k-means: nlist=" + std::to_string(linkage_list.nlist()) +
                    " K1=" + std::to_string(hier2.K1) +
                    " group_size min=" + std::to_string(min_sz) +
                    " max=" + std::to_string(max_sz) +
                    " avg=" + std::to_string(sum_sz / std::max(1, hier2.K1)));
            if (bundle) {
                bundle->hier2_logged = true;
            }
        }
    }

    // ---- HNSW probe index setup (load/build once per session) ----
    std::shared_ptr<ivf::IvfProbeHnswIndex> probe_hnsw_index;
    if (probe_hnsw) {
        if (bundle && bundle->ivf_probe_hnsw) {
            probe_hnsw_index = bundle->ivf_probe_hnsw;
        } else {
            probe_hnsw_index = std::make_shared<ivf::IvfProbeHnswIndex>();
            const std::filesystem::path index_dir =
                ivf::IvfProbeHnswIndex::DefaultIndexDirFromLinkageListDir(std::filesystem::path(linkage_list.dir()));
            const int M_eff = std::max(2, cfg.eval.linkage_ivf_hnsw_M);
            const int efc_eff = std::max(8, cfg.eval.linkage_ivf_hnsw_ef_construction);
            const int build_threads =
                (cfg.runtime.omp_threads > 0) ? cfg.runtime.omp_threads : stlq::OmpMaxThreads();
            bool built = false;
            std::string local_err;
            if (!probe_hnsw_index->LoadOrBuildUnitIP(C_root0, index_dir, M_eff, efc_eff, build_threads, &built, &local_err)) {
                if (err) *err = local_err.empty() ? "IVF probe HNSW: load/build failed." : local_err;
                return false;
            }
            if (bundle) {
                bundle->ivf_probe_hnsw = probe_hnsw_index;
            }
            if (!bundle || !bundle->hnsw_logged) {
                LogInfo(std::string("[ivf_hnsw] ") + (built ? "building index (first time; will be cached)" : "loaded cached index") +
                        " nlist=" + std::to_string(linkage_list.nlist()) +
                        " M=" + std::to_string(M_eff) +
                        " ef_construction=" + std::to_string(efc_eff) +
                        " dir=" + index_dir.string());
                if (bundle) bundle->hnsw_logged = true;
            }
        }
        {
            const int efs_eff = std::max(8, cfg.eval.linkage_ivf_hnsw_ef_search);
            std::string local_err;
            if (!probe_hnsw_index->SetEfSearch(efs_eff, &local_err)) {
                if (err) *err = local_err.empty() ? "IVF probe HNSW: SetEfSearch failed." : local_err;
                return false;
            }
        }
    }

    ColMajorMatrix<float> xCq_coarse_buf;
    if (probe_hier2) {
        xCq_coarse_buf.rows = hier2.K1;
        xCq_coarse_buf.cols = qblk;
        xCq_coarse_buf.data.resize(static_cast<std::size_t>(hier2.K1) * static_cast<std::size_t>(qblk), 0.0f);
    }

    struct FineCandidate {
        float score;
        float raw_dot;
        int id;
    };

    std::vector<std::uint8_t> active_flag(static_cast<std::size_t>(linkage_list.nlist()), 0);
    std::vector<int> active_cids;
    active_cids.reserve(static_cast<std::size_t>(nprobe_cap) * static_cast<std::size_t>(qblk));
    std::vector<float> root_norm2(static_cast<std::size_t>(linkage_list.nlist()), 1.0f);
    for (int cid = 0; cid < linkage_list.nlist(); ++cid) {
        const float* c = C_root0.Col(cid);
        double ss = 0.0;
        for (int r = 0; r < d; ++r) {
            const auto v = static_cast<double>(c[r]);
            ss += v * v;
        }
        root_norm2[static_cast<std::size_t>(cid)] = static_cast<float>(ss);
    }

    const int omp_threads = OmpMaxThreads();
    std::vector<std::vector<int>> active_cids_tls(static_cast<std::size_t>(std::max(1, omp_threads)));
    std::vector<int> q_cids_flat;
    std::vector<int> q_cids_len;
    std::vector<int> cid_to_active;
    std::vector<const LoudsNativeCluster*> active_views;

    for (int q0 = 0; q0 < nquery; q0 += qblk) {
        const int qlen = std::min(qblk, nquery - q0);
        for (int cid : active_cids) active_flag[static_cast<std::size_t>(cid)] = 0;
        active_cids.clear();

        const float* Xq_blk = query_dataset_inmem.Xq.data.data() + static_cast<std::size_t>(q0) * static_cast<std::size_t>(d);
        double t_coarse = 0.0;
        double t_root_small = 0.0;
        double t_one = 0.0;
        const double qt_t0 = omp_get_wtime();
        STLQueryTables qt;
        if (probe_exact) {
            qt = BuildSTLQueryTables(Xq_blk, d, d, qlen, C_root0, meta_root_small, meta_one,
                                       OmpMaxThreads(), &t_coarse, &t_root_small, &t_one);
        } else if (probe_hier2) {
            qt = BuildSTLQueryTablesSkipCoarse(Xq_blk, d, d, qlen, linkage_list.nlist(), meta_root_small,
                                                 meta_one, OmpMaxThreads(), &t_root_small, &t_one);
            xCq_coarse_buf.cols = qlen;
            ScopedBlasThreads blas_scope(OmpMaxThreads());
            const double t0_c = omp_get_wtime();
            GemmRaw(true, false,
                    hier2.K1, qlen, d,
                    1.0f,
                    C_coarse.data.data(), C_coarse.rows,
                    Xq_blk, d,
                    0.0f,
                    xCq_coarse_buf.data.data(), xCq_coarse_buf.rows);
            t_coarse = omp_get_wtime() - t0_c;
        } else {
            // HNSW mode: table GEMMs only; root0 dot products are filled sparsely after HNSW selection.
            qt = BuildSTLQueryTablesSkipCoarse(Xq_blk, d, d, qlen, linkage_list.nlist(), meta_root_small,
                                                 meta_one, OmpMaxThreads(), &t_root_small, &t_one);
        }
        total_core_qt_wall += omp_get_wtime() - qt_t0;
        total_core_qt_gemm_wall += (t_coarse + t_root_small + t_one);
        total_core_qt_coarse_gemm_wall += t_coarse;
        total_core_qt_root_small_gemm_wall += t_root_small;
        total_core_qt_one_gemm_wall += t_one;

        std::vector<TopKHeap> topk;
        topk.reserve(static_cast<std::size_t>(qlen));
        for (int qi = 0; qi < qlen; ++qi) topk.emplace_back(k);

        q_cids_flat.assign(static_cast<std::size_t>(qlen) * static_cast<std::size_t>(nprobe_cap), 0);
        q_cids_len.assign(static_cast<std::size_t>(qlen), 0);

        for (auto& v : active_cids_tls) v.clear();

        const double probe_sel_t0 = omp_get_wtime();
        #pragma omp parallel default(none) shared(qt, xCq_coarse_buf, linkage_list, C_root0, hier2, active_cids_tls, q_cids_flat, q_cids_len, Xq_blk, root_inv_norm_exact, root_inv_norm_hier2, probe_hnsw_index) firstprivate(qlen, nprobe_cap, probe_exact, probe_hier2, probe_hnsw, hier2_topL, d, omp_threads)
        {
            const int tid = omp_get_thread_num();
            std::vector<int> best_ids_local(static_cast<std::size_t>(nprobe_cap), 0);
            std::vector<float> best_scores_local(static_cast<std::size_t>(nprobe_cap), 0.0f);
            std::vector<float> probe_scores_local;
            std::vector<int> top_coarse_ids_local(static_cast<std::size_t>(std::max(1, hier2_topL)), 0);
            std::vector<FineCandidate> fine_cands_local;
            std::vector<int>& active_local = active_cids_tls[static_cast<std::size_t>(std::min(std::max(0, tid), std::max(1, omp_threads) - 1))];
            if (probe_exact) {
                probe_scores_local.assign(static_cast<std::size_t>(linkage_list.nlist()),
                                          -std::numeric_limits<float>::infinity());
            }

            #pragma omp for schedule(static)
            for (int qi = 0; qi < qlen; ++qi) {
                int nprobe = 0;
                if (probe_exact) {
                    const float* scores = qt.xCq_root0.Col(qi);
                    for (int cid = 0; cid < linkage_list.nlist(); ++cid) {
                        probe_scores_local[static_cast<std::size_t>(cid)] =
                            scores[cid] * root_inv_norm_exact[static_cast<std::size_t>(cid)];
                    }
                    nprobe = ivf::SelectTopClustersByScore(
                        probe_scores_local.data(), linkage_list.nlist(), nprobe_cap,
                        best_ids_local.data(), best_scores_local.data());
                } else if (probe_hier2) {
                    const float* dot_coarse = xCq_coarse_buf.Col(qi);
                    std::fill(top_coarse_ids_local.begin(), top_coarse_ids_local.end(), 0);
                    const int got_coarse = ivf::SelectTopCoarseByScore(
                        dot_coarse, nullptr, hier2.K1, hier2_topL,
                        top_coarse_ids_local.data());
                    const float* q = Xq_blk + static_cast<std::size_t>(qi) * static_cast<std::size_t>(d);
                    fine_cands_local.clear();
                    for (int t = 0; t < got_coarse; ++t) {
                        const auto& members = hier2.groups[static_cast<std::size_t>(top_coarse_ids_local[static_cast<std::size_t>(t)])];
                        for (int fid : members) {
                            const float raw_dot = Dot(q, C_root0.Col(fid), d);
                            fine_cands_local.push_back(FineCandidate{
                                raw_dot * root_inv_norm_hier2[static_cast<std::size_t>(fid)],
                                raw_dot,
                                fid});
                        }
                    }
                    const int want = std::min(nprobe_cap, static_cast<int>(fine_cands_local.size()));
                    if (want > 0) {
                        const auto better = [](const FineCandidate& a, const FineCandidate& b) { return a.score > b.score; };
                        if (want < static_cast<int>(fine_cands_local.size())) {
                            std::nth_element(
                                fine_cands_local.begin(), fine_cands_local.begin() + want, fine_cands_local.end(), better);
                        }
                        std::sort(fine_cands_local.begin(), fine_cands_local.begin() + want, better);
                    }
                    nprobe = want;
                    for (int j = 0; j < nprobe; ++j) {
                        best_ids_local[static_cast<std::size_t>(j)] = fine_cands_local[static_cast<std::size_t>(j)].id;
                        best_scores_local[static_cast<std::size_t>(j)] = fine_cands_local[static_cast<std::size_t>(j)].raw_dot;
                    }
                    float* root0_col = qt.xCq_root0.Col(qi);
                    for (int j = 0; j < nprobe; ++j) {
                        root0_col[best_ids_local[static_cast<std::size_t>(j)]] = best_scores_local[static_cast<std::size_t>(j)];
                    }
                } else {
                    // HNSW probe: select centroids by IP over unit-normalized centroids.
                    const float* q = Xq_blk + static_cast<std::size_t>(qi) * static_cast<std::size_t>(d);
                    const int want = std::min(nprobe_cap, linkage_list.nlist());
                    (void)probe_hnsw_index->SearchTopK(q, want,
                                                       best_ids_local.data(),
                                                       /*out_dist=*/nullptr,
                                                       /*err=*/nullptr);
                    float* root0_col = qt.xCq_root0.Col(qi);
                    nprobe = 0;
                    for (int j = 0; j < want; ++j) {
                        const int cid = best_ids_local[static_cast<std::size_t>(j)];
                        if (cid < 0 || cid >= linkage_list.nlist()) continue;
                        const float raw_dot = Dot(q, C_root0.Col(cid), d);
                        best_ids_local[static_cast<std::size_t>(nprobe)] = cid;
                        best_scores_local[static_cast<std::size_t>(nprobe)] = raw_dot;
                        root0_col[cid] = raw_dot;
                        ++nprobe;
                    }
                }

                q_cids_len[static_cast<std::size_t>(qi)] = nprobe;
                for (int t = 0; t < nprobe; ++t) {
                    const int cid = best_ids_local[static_cast<std::size_t>(t)];
                    q_cids_flat[static_cast<std::size_t>(qi) * static_cast<std::size_t>(nprobe_cap) + static_cast<std::size_t>(t)] = cid;
                    active_local.push_back(cid);
                }
            }
        }

        for (const auto& v : active_cids_tls) {
            for (int cid : v) {
                if (active_flag[static_cast<std::size_t>(cid)] == 0) {
                    active_flag[static_cast<std::size_t>(cid)] = 1;
                    active_cids.push_back(cid);
                }
            }
        }
        total_core_probe_sel_wall += omp_get_wtime() - probe_sel_t0;

        cid_to_active.assign(static_cast<std::size_t>(linkage_list.nlist()), -1);
        active_views.assign(active_cids.size(), nullptr);
        for (std::size_t i = 0; i < active_cids.size(); ++i) {
            const LoudsNativeCluster* cl = nullptr;
            if (!LoadCluster(use_coeff_codec, bundle.get(), active_cids[i], cfg, train, &cl, err)) return false;
            active_views[i] = cl;
            cid_to_active[static_cast<std::size_t>(active_cids[i])] = static_cast<int>(i);
        }

        const double scan_t0 = omp_get_wtime();
        #pragma omp parallel default(none) shared(topk, q_cids_len, q_cids_flat, cid_to_active, active_views, qt, offsets_root_small, meta_one, use_coeff_codec, use_norm2_lut, bundle) firstprivate(qlen, nprobe_cap, root_small_total_cols, one_total_cols)
        {
            ScanScratch scratch;
            #pragma omp for schedule(static)
            for (int qi = 0; qi < qlen; ++qi) {
                TopKHeap& heap = topk[static_cast<std::size_t>(qi)];
                const int nprobe = q_cids_len[static_cast<std::size_t>(qi)];
                for (int t = 0; t < nprobe; ++t) {
                    const int cid = q_cids_flat[static_cast<std::size_t>(qi) * static_cast<std::size_t>(nprobe_cap) + static_cast<std::size_t>(t)];
                    const int idx = cid_to_active[static_cast<std::size_t>(cid)];
                    if (idx < 0) continue;
                    const LoudsNativeCluster& cl = *active_views[static_cast<std::size_t>(idx)];
                    const float* lut_centers = bundle->norm2_lut_global
                        ? bundle->global_norm2_lut_centers.data()
                        : cl.norm2_lut.centers.data();
                    if (use_coeff_codec) {
                        if (use_norm2_lut) {
                            ScanOneQueryInt8Lut(
                                cl, cl.norm2_lut.codes_u8.data(), lut_centers,
                                qt.xCq_root0.Col(qi),
                                offsets_root_small.data(), qt.xCq_root_small.Col(qi), root_small_total_cols,
                                meta_one.offsets.data(), qt.xCq_one.Col(qi), one_total_cols,
                                &heap,
                                &scratch);
                        } else {
                            ScanOneQueryInt8(
                                cl,
                                qt.xCq_root0.Col(qi),
                                offsets_root_small.data(), qt.xCq_root_small.Col(qi), root_small_total_cols,
                                meta_one.offsets.data(), qt.xCq_one.Col(qi), one_total_cols,
                                &heap,
                                &scratch);
                        }
                    } else {
                        if (use_norm2_lut) {
                            ScanOneQueryFloatLut(
                                cl, cl.norm2_lut.codes_u8.data(), lut_centers,
                                qt.xCq_root0.Col(qi),
                                offsets_root_small.data(), qt.xCq_root_small.Col(qi),
                                meta_one.offsets.data(), qt.xCq_one.Col(qi),
                                &heap,
                                &scratch);
                        } else {
                            ScanOneQueryFloat(
                                cl,
                                qt.xCq_root0.Col(qi),
                                offsets_root_small.data(), qt.xCq_root_small.Col(qi),
                                meta_one.offsets.data(), qt.xCq_one.Col(qi),
                                &heap,
                                &scratch);
                        }
                    }
                }
            }
        }
        for (int qi = 0; qi < qlen; ++qi) {
            topk[static_cast<std::size_t>(qi)].Finalize(out->dists.Col(q0 + qi), out->indices.Col(q0 + qi));
        }
        total_core_scan_topk_wall += omp_get_wtime() - scan_t0;
    }

    const double core_wall_sec = qt_rotate_wall_sec + total_core_qt_gemm_wall + total_core_probe_sel_wall + total_core_scan_topk_wall;
    if (timing) {
        timing->core_wall_sec = core_wall_sec;
        timing->qt_gemm_wall_sec = total_core_qt_gemm_wall;
        timing->qt_coarse_gemm_wall_sec = total_core_qt_coarse_gemm_wall;
        timing->qt_root_small_gemm_wall_sec = total_core_qt_root_small_gemm_wall;
        timing->qt_one_gemm_wall_sec = total_core_qt_one_gemm_wall;
        timing->qt_other_wall_sec = total_core_qt_wall - total_core_qt_gemm_wall;
        timing->probe_sel_wall_sec = total_core_probe_sel_wall;
        timing->scan_topk_wall_sec = total_core_scan_topk_wall;
        const bool preload_mode = (cfg.eval.linkage_preload_clusters_io_threads >= 0);
        if (preload_mode) {
            timing->louds_wall_sec = bundle->preload_louds_wall_sec;
            timing->louds_cpu_sec = bundle->preload_louds_cpu_sec;
            timing->huffman_wall_sec = bundle->preload_huffman_wall_sec;
            timing->huffman_cpu_sec = bundle->preload_huffman_cpu_sec;
        } else {
            timing->louds_wall_sec = bundle->lazy_louds_wall_sec;
            timing->louds_cpu_sec = bundle->lazy_louds_cpu_sec;
            timing->huffman_wall_sec = bundle->lazy_huffman_wall_sec;
            timing->huffman_cpu_sec = bundle->lazy_huffman_cpu_sec;
        }
        if (session) {
            session->louds_load_mode = (cfg.eval.linkage_preload_clusters_io_threads >= 0) ? "preload-native" : "lazy-native";
        }
    }

    if (!cfg.eval.bench_quiet) {
        if (qt_rotate_wall_sec > 0.0) {
            LogInfo("Disk IVF linkage qt_rotate time: " + std::to_string(qt_rotate_wall_sec) + "s");
        }
        LogInfo("Disk IVF linkage qt_build GEMM time: " + std::to_string(total_core_qt_gemm_wall) +
                "s (coarse=" + std::to_string(total_core_qt_coarse_gemm_wall) +
                "s root_small=" + std::to_string(total_core_qt_root_small_gemm_wall) +
                "s one=" + std::to_string(total_core_qt_one_gemm_wall) + "s)");
        LogInfo("Disk IVF linkage probe_sel time: " + std::to_string(total_core_probe_sel_wall) + "s");
        LogInfo("Disk IVF linkage scan+topk time: " + std::to_string(total_core_scan_topk_wall) + "s");
        LogInfo("Disk IVF linkage core time (qt_rotate+qt_gemm+probe_sel+scan+topk): " + std::to_string(core_wall_sec) + "s");
        if (core_wall_sec > 0.0 && nquery > 0) {
            LogInfo("Disk IVF linkage QPS (core): " + std::to_string(static_cast<double>(nquery) / core_wall_sec));
        }
    }
    const bool want_store = use_coeff_codec ? cfg.eval.linkage_norm2_store_int8
                                          : cfg.eval.linkage_norm2_store_float;
    if (want_store) {
        if (eval::IsDiskNorm2ModeLut(cfg.eval.disk_norm2_mode)) {
            if (eval::IsDiskNorm2ModeLutCluster(cfg.eval.disk_norm2_mode)) {
                std::unordered_map<int, eval::Norm2Lut> lut_cache;
                lut_cache.reserve(bundle->cluster_cache.size());
                for (const auto& [cid, cl] : bundle->cluster_cache) {
                    if (!cl.norm2_lut.codes_u8.empty() && !cl.norm2_lut.centers.empty()) {
                        lut_cache.emplace(cid, cl.norm2_lut);
                    }
                }
                (void)eval::StoreNorm2LutClusterCache(linkage_list, use_coeff_codec, cfg.base.encode.hnorms,
                                                      cfg.eval.disk_norm2_lut_kmeans_niter,
                                                      lut_cache, nullptr);
            }
        } else {
            const char* norm2_filename = use_coeff_codec ? "norm2_int8.f32" : "norm2_float.f32";
            (void)StoreNorm2CacheToDisk(linkage_list, use_coeff_codec, norm2_filename, bundle->cluster_cache);
        }
    }
    return true;
}

}  // namespace

bool EvaluateRecallLinkageIvfFromDiskTimedParentLOUDSNative(const Config& cfg,
                                                          const Dataset& query_dataset_inmem,
                                                          const io::LinkageListReader& linkage_list,
                                                          const TrainResult& train,
                                                          bool use_coeff_codec,
                                                          RecallResult* out,
                                                          DiskLinkageEvalTiming* timing,
                                                          DiskLinkageEvalSession* session,
                                                          std::string* err) {
    if (cfg.runtime.omp_threads > 0) {
        omp_set_num_threads(std::max(1, cfg.runtime.omp_threads));
    }
    return EvalImpl(cfg, use_coeff_codec, query_dataset_inmem, linkage_list, train, out, timing, session, err);
}

bool PrepareRecallLinkageIvfDiskSessionParentLOUDSNative(const Config& cfg,
                                                       const io::LinkageListReader& linkage_list,
                                                       const TrainResult& train,
                                                       bool use_coeff_codec,
                                                       DiskLinkageEvalTiming* timing,
                                                       DiskLinkageEvalSession* session,
                                                       std::string* err) {
    if (cfg.runtime.omp_threads > 0) {
        omp_set_num_threads(std::max(1, cfg.runtime.omp_threads));
    }
    const bool want_store = use_coeff_codec ? cfg.eval.linkage_norm2_store_int8
                                           : cfg.eval.linkage_norm2_store_float;
    if (want_store) {
        if (eval::IsDiskNorm2ModeLut(cfg.eval.disk_norm2_mode)) {
            const bool use_global_lut = eval::IsDiskNorm2ModeLutGlobal(cfg.eval.disk_norm2_mode);
            std::string lut_err;
            const auto load_res = use_global_lut
                ? eval::CheckNorm2LutCache(linkage_list, use_coeff_codec,
                                           cfg.base.encode.hnorms,
                                           cfg.eval.disk_norm2_lut_kmeans_niter,
                                           cfg.eval.disk_norm2_mode,
                                           cfg.eval.disk_norm2_lut_log_alpha,
                                           cfg.eval.disk_norm2_lut_piecewise_p1,
                                           cfg.eval.disk_norm2_lut_piecewise_p2,
                                           cfg.eval.disk_norm2_lut_piecewise_count_weight,
                                           cfg.eval.disk_norm2_lut_piecewise_range_weight,
                                           &lut_err)
                : eval::CheckNorm2LutClusterCache(linkage_list, use_coeff_codec,
                                                  cfg.base.encode.hnorms,
                                                  cfg.eval.disk_norm2_lut_kmeans_niter,
                                                  &lut_err);
            if (load_res != eval::Norm2LutDiskLoadResult::kLoaded) {
                if (load_res == eval::Norm2LutDiskLoadResult::kInvalid) {
                    LogWarn(std::string("Rebuilding stale ") +
                            (use_global_lut ? "global" : "cluster") +
                            " norm2 LUT cache (" +
                            (lut_err.empty() ? std::string("invalid cache") : lut_err) + ").");
                } else {
                    LogInfo(std::string("Precomputing full ") +
                            (use_global_lut ? "global" : "cluster") +
                            " norm2 LUT cache ...");
                }
                if (!BuildAndStoreFullNorm2LutCacheNative(use_coeff_codec, cfg, linkage_list, train, err)) {
                    return false;
                }
            }
        } else {
            std::unordered_map<int, std::vector<float>> ignored;
            std::string load_err;
            if (LoadNorm2CacheFromDisk(linkage_list,
                                       use_coeff_codec,
                                       use_coeff_codec ? "norm2_int8.f32" : "norm2_float.f32",
                                       &ignored,
                                       nullptr,
                                       &load_err) != Norm2DiskLoadResult::kLoaded) {
                LogInfo("Precomputing full norm2 cache for " +
                        std::string(use_coeff_codec ? "norm2_int8.f32" : "norm2_float.f32") + " ...");
                if (!BuildAndStoreFullNorm2CacheNative(use_coeff_codec, cfg, linkage_list, train, err)) {
                    return false;
                }
            }
        }
    }
    std::shared_ptr<LoudsNativeBundle> bundle;
    return EnsureLoudsNativeBundle(use_coeff_codec, cfg, linkage_list, train, timing, session, &bundle, err);
}

}  // namespace stlq
