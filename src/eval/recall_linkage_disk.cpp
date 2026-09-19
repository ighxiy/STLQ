#include "stlq/eval/recall_linkage_disk.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
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

#include "stlq/core/blas.h"
#include "stlq/core/threading.h"
#include "stlq/coeff/bit_io.h"
#include "stlq/coeff/huffman_canonical.h"
#include "stlq/coeff/span.h"
#include "stlq/pipeline/large_store_hash.h"
#include "stlq/eval/linkage_cluster_provider.h"
#include "stlq/eval/linkage_norm_provider.h"
#include "stlq/eval/norm2_lut.h"
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
#include "stlq/eval/linkage_norm_provider_cuda.h"
#include "stlq/eval/recall_linkage_scan_cuda.h"
#endif
#include "stlq/eval/query_table_builder.h"
#include "stlq/io/linkage_coeff_codec_store.h"
#include "stlq/io/linkage_list_store.h"
#include "stlq/ivf/cluster_select.h"
#include "stlq/ivf/ivf_probe_hnsw_index.h"
#include "stlq/common/logger.h"
#include "stlq/quantizer/codebook_meta.h"
#include "stlq/common/timer.h"

#include <omp.h>

#include "stlq/common/types.h"

namespace stlq
{
    namespace
    {
        static std::unique_ptr<stlq::eval::IClusterNormProvider> MakeNormProviderOwned(
            const Config& cfg,
            bool use_coeff_codec,
            const ColMajorMatrix<float>& C_root0,
            const CodebookMeta& meta_root_small,
            const CodebookMeta& meta_one,
            bool profile_timing);

        std::filesystem::path Norm2CacheHashPath(const std::filesystem::path& norm2_path) {
            return norm2_path.string() + ".hash.u64";
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
            }
            else {
                try {
                    v = static_cast<std::uint64_t>(std::stoull(t));
                }
                catch (...) {
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

        static std::filesystem::path DefaultHier2SplitCacheDirFromLinkageListDir(
            const std::filesystem::path& linkage_list_dir) {
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
            auto get = [&](const char* k) -> std::string
            {
                auto it = meta.find(k);
                return (it == meta.end()) ? std::string() : it->second;
            };

            auto parse_i32 = [](const std::string& s, int* out_i) -> bool
            {
                if (!out_i) return false;
                try {
                    *out_i = std::stoi(TrimStr(s));
                    return true;
                }
                catch (...) {
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

        std::filesystem::path Norm2CacheCountPath(const std::filesystem::path& norm2_path) {
            return norm2_path.string() + ".count.u64";
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

        bool ValidateNorm2CacheHash(const io::LinkageListReader& linkage_list,
                                    bool use_coeff_codec,
                                    const std::filesystem::path& norm2_path,
                                    std::string* reason) {
            std::uint64_t source_hash = 0;
            if (!ReadSourceHashForNorm2Cache(linkage_list, use_coeff_codec, &source_hash)) {
                if (reason) {
                    *reason = use_coeff_codec
                                  ? "missing linkage_list/hash.u64 or coeff_hash.u64"
                                  : "missing linkage_list/hash.u64";
                }
                return false;
            }

            std::uint64_t cache_hash = 0;
            const auto cache_hash_path = Norm2CacheHashPath(norm2_path);
            if (!app::ReadU64File(cache_hash_path.string(), &cache_hash)) {
                if (reason) *reason = "missing " + cache_hash_path.filename().string();
                return false;
            }
            if (cache_hash != source_hash) {
                if (reason)
                    *reason = use_coeff_codec
                                  ? "norm2 source hash mismatch (linkage_list/hash.u64 + coeff_hash.u64)"
                                  : "store hash mismatch";
                return false;
            }
            std::uint64_t cache_count = 0;
            if (!app::ReadU64File(Norm2CacheCountPath(norm2_path).string(), &cache_count)) {
                if (reason) *reason = "missing " + Norm2CacheCountPath(norm2_path).filename().string();
                return false;
            }
            const std::uint64_t expected_count = CountNonEmptyRealClusters(linkage_list);
            if (cache_count != expected_count) {
                if (reason) *reason = "incomplete cluster coverage";
                return false;
            }
            return true;
        }

        bool CheckFloatNorm2Cache(const io::LinkageListReader& linkage_list,
                                  bool use_coeff_codec,
                                  const std::filesystem::path& norm2_path,
                                  std::string* reason) {
            if (!ValidateNorm2CacheHash(linkage_list, use_coeff_codec, norm2_path, reason)) {
                return false;
            }
            std::error_code fec;
            const auto actual_bytes = std::filesystem::file_size(norm2_path, fec);
            const std::uint64_t expected_bytes = linkage_list.total_real() * sizeof(float);
            if (fec || actual_bytes != expected_bytes ||
                linkage_list.real_offsets().size() != static_cast<std::size_t>(linkage_list.nlist() + 1)) {
                if (reason) *reason = "unexpected norm2 payload size";
                return false;
            }
            return true;
        }

        bool LoadFlatFloatNorm2Cache(const io::LinkageListReader& linkage_list,
                                     bool use_coeff_codec,
                                     const std::filesystem::path& norm2_path,
                                     std::vector<float>* flat_norm2,
                                     std::string* err) {
            if (!flat_norm2) {
                if (err) *err = "LoadFlatFloatNorm2Cache: null output.";
                return false;
            }
            std::string reason;
            if (!CheckFloatNorm2Cache(linkage_list, use_coeff_codec, norm2_path, &reason)) {
                if (err) *err = reason.empty() ? "invalid float norm2 cache" : reason;
                return false;
            }
            std::ifstream fin(norm2_path, std::ios::binary);
            if (!fin) {
                if (err) *err = "LoadFlatFloatNorm2Cache: failed to open " + norm2_path.string();
                return false;
            }
            flat_norm2->assign(static_cast<std::size_t>(linkage_list.total_real()), 0.0f);
            fin.read(reinterpret_cast<char*>(flat_norm2->data()),
                     static_cast<std::streamsize>(flat_norm2->size() * sizeof(float)));
            if (!fin) {
                if (err) *err = "LoadFlatFloatNorm2Cache: failed to read " + norm2_path.string();
                flat_norm2->clear();
                return false;
            }
            return true;
        }

        int ResolveNorm2WorkerThreads(const io::LinkageListReader& linkage_list, int requested_threads) {
            const int omp_default = GetOmpDefaultThreads();
            return std::max(
                1,
                std::min((requested_threads == 0) ? std::max(1, omp_default) : std::max(1, requested_threads),
                         std::max(1, linkage_list.nlist())));
        }

        bool BuildNorm2LutCacheFromFlat(const io::LinkageListReader& linkage_list,
                                        eval::IClusterNormProvider* norm_provider,
                                        bool use_global_lut,
                                        int hnorms,
                                        int kmeans_niter,
                                        const std::string& disk_norm2_mode,
                                        double log1p_alpha,
                                        double piecewise_p1,
                                        double piecewise_p2,
                                        double piecewise_count_weight,
                                        double piecewise_range_weight,
                                        int worker_threads,
                                        const std::vector<float>& flat_norm2,
                                        eval::Norm2Lut* global_lut_cache,
                                        std::unordered_map<int, eval::Norm2Lut>* cluster_lut_cache,
                                        std::string* err) {
            if (!norm_provider) {
                if (err) *err = "BuildNorm2LutCacheFromFlat: null norm provider.";
                return false;
            }
            if (use_global_lut) {
                if (!global_lut_cache) {
                    if (err) *err = "BuildNorm2LutCacheFromFlat: null global LUT output.";
                    return false;
                }
                return eval::BuildNorm2LutWithMode(flat_norm2.data(),
                                                   static_cast<int>(flat_norm2.size()),
                                                   hnorms,
                                                   kmeans_niter,
                                                   disk_norm2_mode,
                                                   log1p_alpha,
                                                   piecewise_p1,
                                                   piecewise_p2,
                                                   piecewise_count_weight,
                                                   piecewise_range_weight,
                                                   global_lut_cache,
                                                   err);
            }
            if (!cluster_lut_cache) {
                if (err) *err = "BuildNorm2LutCacheFromFlat: null cluster LUT output.";
                return false;
            }
            cluster_lut_cache->clear();
            cluster_lut_cache->reserve(static_cast<std::size_t>(linkage_list.nlist()));
            const auto& real_offs = linkage_list.real_offsets();
            const int nlist = linkage_list.nlist();

            const int n_threads = ResolveNorm2WorkerThreads(linkage_list, worker_threads);
            std::vector<std::unique_ptr<eval::IClusterNormProvider>> provider_clones;
            std::vector<eval::IClusterNormProvider*> thread_providers;
            provider_clones.reserve(static_cast<std::size_t>(n_threads));
            for (int t = 0; t < n_threads; ++t) {
                auto clone = norm_provider->CloneForParallelPrecompute();
                if (!clone) {
                    provider_clones.clear();
                    thread_providers.clear();
                    break;
                }
                thread_providers.push_back(clone.get());
                provider_clones.push_back(std::move(clone));
            }

            if (thread_providers.empty()) {
                for (int cid = 0; cid < nlist; ++cid) {
                    const auto lo = real_offs[static_cast<std::size_t>(cid)];
                    const auto hi = real_offs[static_cast<std::size_t>(cid + 1)];
                    const auto nr = hi - lo;
                    if (nr <= 0) continue;
                    eval::Norm2Lut lut;
                    if (!norm_provider->ComputeNorm2LutFromHost(flat_norm2.data() + static_cast<std::size_t>(lo),
                                                                static_cast<int>(nr),
                                                                hnorms,
                                                                kmeans_niter,
                                                                &lut,
                                                                err)) {
                        return false;
                    }
                    cluster_lut_cache->emplace(cid, std::move(lut));
                }
                return true;
            }

            struct ThreadResult
            {
                std::vector<std::pair<int, eval::Norm2Lut>> luts;
                std::string err;
                bool ok = true;
            };
            std::vector<ThreadResult> results(static_cast<std::size_t>(n_threads));
            std::vector<std::thread> threads;
            threads.reserve(static_cast<std::size_t>(n_threads));
            for (int t = 0; t < n_threads; ++t) {
                threads.emplace_back([&, t]()
                {
                    auto& res = results[static_cast<std::size_t>(t)];
                    for (int cid = t; cid < nlist; cid += n_threads) {
                        const auto lo = real_offs[static_cast<std::size_t>(cid)];
                        const auto hi = real_offs[static_cast<std::size_t>(cid + 1)];
                        const auto nr = hi - lo;
                        if (nr <= 0) continue;
                        eval::Norm2Lut lut;
                        if (!thread_providers[static_cast<std::size_t>(t)]->ComputeNorm2LutFromHost(
                            flat_norm2.data() + static_cast<std::size_t>(lo),
                            static_cast<int>(nr),
                            hnorms,
                            kmeans_niter,
                            &lut,
                            &res.err)) {
                            res.ok = false;
                            return;
                        }
                        res.luts.emplace_back(cid, std::move(lut));
                    }
                });
            }
            for (auto& th : threads) th.join();
            for (auto& res : results) {
                if (!res.ok) {
                    if (err) *err = res.err;
                    return false;
                }
                for (auto& entry : res.luts) {
                    cluster_lut_cache->emplace(entry.first, std::move(entry.second));
                }
            }
            return true;
        }

        bool StoreNorm2CacheMap(const io::LinkageListReader& linkage_list,
                                bool use_coeff_codec,
                                const std::filesystem::path& out_path,
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
            if (static_cast<std::uint64_t>(stored) != expected_clusters) {
                if (err) {
                    *err = "StoreNorm2CacheMap: incomplete cluster coverage (" +
                        std::to_string(stored) + "/" + std::to_string(expected_clusters) + ").";
                }
                return false;
            }

            std::ofstream fout(out_path, std::ios::binary);
            if (!fout) {
                if (err) *err = "StoreNorm2CacheMap: failed to open " + out_path.string();
                return false;
            }
            fout.write(reinterpret_cast<const char*>(flat.data()),
                       static_cast<std::streamsize>(flat.size() * sizeof(float)));
            if (!fout) {
                if (err) *err = "StoreNorm2CacheMap: failed to write " + out_path.string();
                return false;
            }

            std::uint64_t source_hash = 0;
            if (ReadSourceHashForNorm2Cache(linkage_list, use_coeff_codec, &source_hash)) {
                std::string hash_err;
                if (!app::WriteU64FileHex(Norm2CacheHashPath(out_path).string(), source_hash, &hash_err)) {
                    if (err) *err = hash_err;
                    return false;
                }
                std::string count_err;
                if (!app::WriteU64FileHex(Norm2CacheCountPath(out_path).string(), expected_clusters, &count_err)) {
                    if (err) *err = count_err;
                    return false;
                }
            }
            return true;
        }

        bool BuildAndStoreFullNorm2Cache(const Config& cfg,
                                         const io::LinkageListReader& linkage_list,
                                         const ColMajorMatrix<float>& C_root0,
                                         const CodebookMeta& meta_root_small,
                                         const CodebookMeta& meta_one,
                                         bool use_coeff_codec,
                                         const std::filesystem::path& out_path,
                                         std::string* err) {
            io::LinkageCoeffCodecReader coeff_reader;
            const io::LinkageCoeffCodecReader* coeff_reader_ptr = nullptr;
            if (use_coeff_codec) {
                std::string local_err;
                if (!coeff_reader.Open(linkage_list.dir(), &local_err)) {
                    if (err)
                        *err = local_err.empty()
                                   ? "BuildAndStoreFullNorm2Cache: missing coeff codec store."
                                   : local_err;
                    return false;
                }
                coeff_reader_ptr = &coeff_reader;
            }

            auto norm_owned = MakeNormProviderOwned(cfg, use_coeff_codec, C_root0, meta_root_small,
                                                    meta_one, /*profile_timing=*/false);
            eval::ClusterProvider provider;
            if (!provider.Open(linkage_list,
                               coeff_reader_ptr,
                               norm_owned.get(),
                               use_coeff_codec,
                               cfg.eval.linkage_parent_louds_enable,
                               static_cast<std::uint32_t>(std::max(1, cfg.eval.parent_louds_select_stride)),
                               static_cast<std::uint32_t>(std::max(1, cfg.eval.parent_louds_rank_words_per_super_log2)),
                               cfg.eval.parent_louds_build_indices,
                               cfg.eval.linkage_parent_adaptive_u16_cache,
                               /*use_norm2_lut=*/false,
                               /*use_norm2_lut_global=*/false,
                               cfg.base.encode.hnorms,
                               cfg.eval.disk_norm2_lut_kmeans_niter,
                               /*profile_prep_stats=*/false,
                               err)) {
                return false;
            }

            std::unordered_map<int, std::vector<float>> full_norm_cache;
            const int norm_threads = (cfg.eval.linkage_preload_clusters_io_threads >= 0)
                                         ? cfg.eval.linkage_preload_clusters_io_threads
                                         : 0;
            if (!provider.PrecomputeAllNorm2(&full_norm_cache, norm_threads, err)) {
                return false;
            }
            return StoreNorm2CacheMap(linkage_list, use_coeff_codec, out_path, full_norm_cache, err);
        }

        bool BuildAndStoreFullNorm2LutCache(const Config& cfg,
                                            const io::LinkageListReader& linkage_list,
                                            const ColMajorMatrix<float>& C_root0,
                                            const CodebookMeta& meta_root_small,
                                            const CodebookMeta& meta_one,
                                            bool use_coeff_codec,
                                            std::string* err) {
            io::LinkageCoeffCodecReader coeff_reader;
            const io::LinkageCoeffCodecReader* coeff_reader_ptr = nullptr;
            if (use_coeff_codec) {
                std::string local_err;
                if (!coeff_reader.Open(linkage_list.dir(), &local_err)) {
                    if (err)
                        *err = local_err.empty()
                                   ? "BuildAndStoreFullNorm2LutCache: missing coeff codec store."
                                   : local_err;
                    return false;
                }
                coeff_reader_ptr = &coeff_reader;
            }

            auto norm_owned = MakeNormProviderOwned(cfg, use_coeff_codec, C_root0, meta_root_small,
                                                    meta_one, /*profile_timing=*/false);
            const bool use_global_lut = eval::IsDiskNorm2ModeLutGlobal(cfg.eval.disk_norm2_mode);
            const std::filesystem::path norm2_path =
                std::filesystem::path(linkage_list.dir()) / (use_coeff_codec ? "norm2_int8.f32" : "norm2_float.f32");
            const int lut_worker_threads = (cfg.eval.linkage_preload_clusters_io_threads >= 0)
                                               ? cfg.eval.linkage_preload_clusters_io_threads
                                               : 0;
            double norm_build_sec = 0.0;
            double lut_build_sec = 0.0;
            eval::Norm2Lut global_lut_cache;
            std::unordered_map<int, eval::Norm2Lut> cluster_lut_cache;
            std::vector<float> flat_norm2;
            std::string float_cache_err;
            if (LoadFlatFloatNorm2Cache(linkage_list, use_coeff_codec, norm2_path, &flat_norm2, &float_cache_err)) {
                LogInfo("Reusing validated float norm2 cache " + norm2_path.filename().string() +
                    " to build " + std::string(use_global_lut ? "global" : "cluster") + " norm2 LUT cache.");
                Timer t_lut;
                if (!BuildNorm2LutCacheFromFlat(linkage_list,
                                                norm_owned.get(),
                                                use_global_lut,
                                                cfg.base.encode.hnorms,
                                                cfg.eval.disk_norm2_lut_kmeans_niter,
                                                cfg.eval.disk_norm2_mode,
                                                cfg.eval.disk_norm2_lut_log_alpha,
                                                cfg.eval.disk_norm2_lut_piecewise_p1,
                                                cfg.eval.disk_norm2_lut_piecewise_p2,
                                                cfg.eval.disk_norm2_lut_piecewise_count_weight,
                                                cfg.eval.disk_norm2_lut_piecewise_range_weight,
                                                lut_worker_threads,
                                                flat_norm2,
                                                &global_lut_cache,
                                                &cluster_lut_cache,
                                                err)) {
                    return false;
                }
                lut_build_sec = t_lut.ElapsedSeconds();
                LogInfo("Full norm2 LUT cache build timing(s): norm2=" +
                    std::to_string(norm_build_sec) + " lut_kmeans=" + std::to_string(lut_build_sec));
                if (use_global_lut) {
                    return eval::StoreNorm2LutCache(linkage_list, use_coeff_codec, cfg.base.encode.hnorms,
                                                    cfg.eval.disk_norm2_lut_kmeans_niter,
                                                    cfg.eval.disk_norm2_mode,
                                                    cfg.eval.disk_norm2_lut_log_alpha,
                                                    cfg.eval.disk_norm2_lut_piecewise_p1,
                                                    cfg.eval.disk_norm2_lut_piecewise_p2,
                                                    cfg.eval.disk_norm2_lut_piecewise_count_weight,
                                                    cfg.eval.disk_norm2_lut_piecewise_range_weight,
                                                    global_lut_cache, err);
                }
                return eval::StoreNorm2LutClusterCache(linkage_list, use_coeff_codec, cfg.base.encode.hnorms,
                                                       cfg.eval.disk_norm2_lut_kmeans_niter, cluster_lut_cache, err);
            }
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
            if (!use_global_lut) {
                if (auto* cuda_norm = dynamic_cast<eval::cuda::LinkageNormProviderLookupCuda*>(norm_owned.get())) {
                    eval::ClusterProvider provider;
                    if (!provider.Open(linkage_list,
                                       coeff_reader_ptr,
                                       norm_owned.get(),
                                       use_coeff_codec,
                                       cfg.eval.linkage_parent_louds_enable,
                                       static_cast<std::uint32_t>(std::max(1, cfg.eval.parent_louds_select_stride)),
                                       static_cast<std::uint32_t>(std::max(
                                           1, cfg.eval.parent_louds_rank_words_per_super_log2)),
                                       cfg.eval.parent_louds_build_indices,
                                       cfg.eval.linkage_parent_adaptive_u16_cache,
                                       /*use_norm2_lut=*/true,
                                       /*use_norm2_lut_global=*/false,
                                       cfg.base.encode.hnorms,
                                       cfg.eval.disk_norm2_lut_kmeans_niter,
                                       /*profile_prep_stats=*/false,
                                       err)) {
                        return false;
                    }
                    cluster_lut_cache.reserve(static_cast<std::size_t>(linkage_list.nlist()));
                    for (int cid = 0; cid < linkage_list.nlist(); ++cid) {
                        eval::ClusterView cv;
                        eval::ClusterProvider::PrepStats prep{};
                        std::string local_err;
                        if (!provider.GetCluster(cid, &cv, &prep, &local_err)) {
                            if (err) *err = local_err;
                            return false;
                        }
                        if (cv.n_real <= 0 || !cv.r_norm2_lut_u8 || !cv.r_norm2_lut_centers || cv.r_norm2_lut_size <=
                            0) {
                            continue;
                        }
                        const auto& stats = cuda_norm->last_stats();
                        norm_build_sec += stats.cpu_c0dot_sec + stats.h2d_sec + stats.kernel_sec + stats.d2h_sec;
                        lut_build_sec += stats.lut_kmeans_sec;
                    }
                    cluster_lut_cache = provider.ExportNorm2LutCache();
                }
            }
#endif
            if (use_global_lut || cluster_lut_cache.empty()) {
                eval::ClusterProvider provider;
                if (!provider.Open(linkage_list,
                                   coeff_reader_ptr,
                                   norm_owned.get(),
                                   use_coeff_codec,
                                   cfg.eval.linkage_parent_louds_enable,
                                   static_cast<std::uint32_t>(std::max(1, cfg.eval.parent_louds_select_stride)),
                                   static_cast<std::uint32_t>(std::max(
                                       1, cfg.eval.parent_louds_rank_words_per_super_log2)),
                                   cfg.eval.parent_louds_build_indices,
                                   cfg.eval.linkage_parent_adaptive_u16_cache,
                                   /*use_norm2_lut=*/false,
                                   /*use_norm2_lut_global=*/false,
                                   cfg.base.encode.hnorms,
                                   cfg.eval.disk_norm2_lut_kmeans_niter,
                                   /*profile_prep_stats=*/false,
                                   err)) {
                    return false;
                }

                std::unordered_map<int, std::vector<float>> full_norm_cache;
                const int norm_threads = (cfg.eval.linkage_preload_clusters_io_threads >= 0)
                                             ? cfg.eval.linkage_preload_clusters_io_threads
                                             : 0;
                Timer t_norm;
                if (!provider.PrecomputeAllNorm2(&full_norm_cache, norm_threads, err)) {
                    return false;
                }
                norm_build_sec = t_norm.ElapsedSeconds();

                Timer t_lut;
                if (use_global_lut) {
                    const auto& real_offs = linkage_list.real_offsets();
                    std::vector<float> flat_norm2(static_cast<std::size_t>(linkage_list.total_real()), 0.0f);
                    for (const auto& [cid, r_norm2] : full_norm_cache) {
                        if (cid < 0 || cid >= linkage_list.nlist()) continue;
                        const auto lo = real_offs[static_cast<std::size_t>(cid)];
                        const auto hi = real_offs[static_cast<std::size_t>(cid + 1)];
                        const auto nr = hi - lo;
                        if (r_norm2.size() != static_cast<std::size_t>(nr)) continue;
                        std::memcpy(flat_norm2.data() + static_cast<std::size_t>(lo),
                                    r_norm2.data(),
                                    static_cast<std::size_t>(nr) * sizeof(float));
                    }
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                    if (auto* cuda_norm = dynamic_cast<eval::cuda::LinkageNormProviderLookupCuda*>(norm_owned.get())) {
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
                                                     &global_lut_cache,
                                                     err)) {
                        return false;
                    }
                    lut_build_sec = t_lut.ElapsedSeconds();
                }
                else {
                    cluster_lut_cache.reserve(full_norm_cache.size());
                    for (auto& [cid, r_norm2] : full_norm_cache) {
                        eval::Norm2Lut lut;
                        if (!eval::BuildNorm2Lut(r_norm2.data(),
                                                 static_cast<int>(r_norm2.size()),
                                                 cfg.base.encode.hnorms,
                                                 cfg.eval.disk_norm2_lut_kmeans_niter,
                                                 &lut,
                                                 err)) {
                            return false;
                        }
                        cluster_lut_cache.emplace(cid, std::move(lut));
                    }
                    lut_build_sec = t_lut.ElapsedSeconds();
                }
            }
            LogInfo("Full norm2 LUT cache build timing(s): norm2=" +
                std::to_string(norm_build_sec) + " lut_kmeans=" + std::to_string(lut_build_sec));
            if (use_global_lut) {
                return eval::StoreNorm2LutCache(linkage_list, use_coeff_codec, cfg.base.encode.hnorms,
                                                cfg.eval.disk_norm2_lut_kmeans_niter,
                                                cfg.eval.disk_norm2_mode,
                                                cfg.eval.disk_norm2_lut_log_alpha,
                                                cfg.eval.disk_norm2_lut_piecewise_p1,
                                                cfg.eval.disk_norm2_lut_piecewise_p2,
                                                cfg.eval.disk_norm2_lut_piecewise_count_weight,
                                                cfg.eval.disk_norm2_lut_piecewise_range_weight,
                                                global_lut_cache, err);
            }
            return eval::StoreNorm2LutClusterCache(linkage_list, use_coeff_codec, cfg.base.encode.hnorms,
                                                   cfg.eval.disk_norm2_lut_kmeans_niter, cluster_lut_cache, err);
        }

        // Deterministic ordering: (dist, id).
        // Better = smaller dist; ties => smaller id.
        struct Candidate
        {
            float dist = 0.0f;
            std::int32_t id = -1;
        };

        STLQ_ALWAYS_INLINE bool CandLess(const Candidate& a, const Candidate& b) {
            if (a.dist < b.dist) return true;
            if (a.dist > b.dist) return false;
            return a.id < b.id;
        }

        struct TopKHeap
        {
            int k = 0;
            std::vector<Candidate> heap; // max-heap by CandLess (so heap[0] is current "worst")
            bool heapified = false;

            explicit TopKHeap(int k_in) : k(std::max(1, k_in)) {
                heap.reserve(static_cast<std::size_t>(k));
            }

            STLQ_ALWAYS_INLINE void SiftDown(int i) {
                // "hole" sift-down: fewer moves than swap-per-level.
                const int n = static_cast<int>(heap.size());
                Candidate x = heap[static_cast<std::size_t>(i)];
                while (true) {
                    const int l = i * 2 + 1;
                    if (l >= n) break;
                    int c = l;
                    const int r = l + 1;
                    if (r < n && CandLess(heap[static_cast<std::size_t>(c)],
                                          heap[static_cast<std::size_t>(r)])) {
                        c = r;
                    }
                    // If x is not better than the worse child, heap property holds.
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
                    if (static_cast<int>(heap.size()) == k) {
                        Heapify();
                    }
                    return;
                }
                if (!heapified) {
                    Heapify();
                }
                const Candidate& worst = heap[0];
                if (x.dist > worst.dist || (x.dist == worst.dist && x.id >= worst.id)) return;
                heap[0] = x;
                SiftDown(0);
            }

            // Assumes: heap.size()==k, heapified==true, and (dist,id) is strictly better than current worst.
            STLQ_ALWAYS_INLINE void PushAssumeBetter(float dist, std::int32_t id) {
                heap[0] = Candidate{dist, id};
                SiftDown(0);
            }

            void Finalize(float* out_dists, int* out_ids) {
                std::sort(heap.begin(), heap.end(), [](const Candidate& a, const Candidate& b)
                {
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

        struct ScanScratch
        {
            std::vector<float> dot_all; // size nc (n_real+n_virt)
            std::vector<float> root_small_scaled; // size root_small_total_cols
            std::vector<float> one_scaled; // size one_total_cols
            std::vector<float> dist_tmp; // size n_real (profile-only)
        };

        // In the large-scale two-codebook pipeline, code0_one.bin stores C_one[0] codes for linkage nodes
        // and is always uint8 on this eval path.
        STLQ_ALWAYS_INLINE std::uint8_t ReadCode0One(const std::uint8_t* base, int pos) {
            return base[pos];
        }

        struct ScanKernelTiming
        {
            double t_scale_tables = 0.0;
            double t_roots_init = 0.0;
            double t_roots_layers = 0.0;
            double t_linkage = 0.0;
            double t_push = 0.0;
            double t_push_dist = 0.0;
            double t_push_heap = 0.0;
        };

        // Scan kernel signature for the int8 coeff codec path.
        using ScanCoeffFn = void (*)(const eval::ClusterView&,
                                     const float* __restrict,
                                     const int* __restrict,
                                     const float* __restrict,
                                     int,
                                     const int* __restrict,
                                     const float* __restrict,
                                     int,
                                     TopKHeap* __restrict,
                                     ScanScratch* __restrict,
                                     ScanKernelTiming* __restrict);

        struct ScanTopKProfileSums
        {
            bool enabled = false;

            double* total_scan_kernel = nullptr;
            double* total_topk_finalize = nullptr;
            double* total_topk_finalize_worker = nullptr;

            double* total_scan_scale_tables = nullptr;
            double* total_scan_roots_init = nullptr;
            double* total_scan_roots_layers = nullptr;
            double* total_scan_linkage = nullptr;
            double* total_scan_push = nullptr;
            double* total_scan_push_dist = nullptr;
            double* total_scan_push_heap = nullptr;

            // GPU scan (int8 coeff codec only).
            double* total_gpu_scan_pack_h2d = nullptr;
            double* total_gpu_scan_host_pack = nullptr;
            double* total_gpu_scan_alloc = nullptr;
            double* total_gpu_scan_tables_h2d = nullptr;
            double* total_gpu_scan_kernel = nullptr;
            double* total_gpu_scan_k_roots = nullptr;
            double* total_gpu_scan_k_depth = nullptr;
            double* total_gpu_scan_k_topk = nullptr;
            double* total_gpu_scan_out_d2h = nullptr;
            double* total_gpu_scan_stream_sync = nullptr;
            double* total_gpu_scan_cpu_fallback = nullptr;
            int* total_gpu_scan_tasks_total = nullptr;
            int* total_gpu_scan_tasks_gpu = nullptr;
            int* total_gpu_scan_clusters_total = nullptr;
            int* total_gpu_scan_clusters_gpu = nullptr;
            int* total_gpu_scan_kernel_max_nc = nullptr;
            int* total_gpu_scan_cache_slots = nullptr;
            int* total_gpu_scan_cache_calls = nullptr;
            int* total_gpu_scan_cache_enabled_calls = nullptr;
            int* total_gpu_scan_cache_hits = nullptr;
            int* total_gpu_scan_cache_misses = nullptr;
            int* total_gpu_scan_cache_upload_clusters = nullptr;
            std::uint64_t* total_gpu_scan_cache_upload_bytes = nullptr;
            std::uint64_t* total_gpu_scan_task_cluster_idx_bytes = nullptr;
            std::uint64_t* total_gpu_scan_out_d2h_bytes = nullptr;
            bool* gpu_scan_failed_any = nullptr;
            std::string* gpu_scan_first_err = nullptr;
        };

        static std::unique_ptr<stlq::eval::IClusterNormProvider> MakeNormProviderOwned(
            const Config& cfg,
            bool use_coeff_codec,
            const ColMajorMatrix<float>& C_root0,
            const CodebookMeta& meta_root_small,
            const CodebookMeta& meta_one,
            bool profile_timing) {
            auto norm_cpu_owned = std::make_unique<eval::LinkageNormProviderLookup>(C_root0, meta_root_small, meta_one);

            (void)use_coeff_codec;
            (void)cfg;
            (void)profile_timing;

#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
            if (cfg.eval.linkage_gpu_norm_enable && cfg.runtime.use_cuda) {
                const int device_max_nc = eval::cuda::DiskLinkageGpuScanMaxNcSupported();
                const int effective_max_nc =
                    (device_max_nc > 0)
                        ? std::min(cfg.eval.linkage_gpu_scan_max_nc, device_max_nc)
                        : cfg.eval.linkage_gpu_scan_max_nc;
                return std::make_unique<eval::cuda::LinkageNormProviderLookupCuda>(
                    C_root0,
                    meta_root_small,
                    meta_one,
                    effective_max_nc,
                    /*profile_breakdown=*/profile_timing);
            }
#endif

            return norm_cpu_owned;
        }

        STLQ_ALWAYS_INLINE void CopyScalePackedTable(float* __restrict dst,
                                                     const float* __restrict src,
                                                     const int* __restrict offsets,
                                                     int total_cols,
                                                     const float* __restrict scales,
                                                     int m,
                                                     int start_layer) {
            // Scaled copy (avoids an extra memcpy + in-place multiply).
            // Layers before start_layer are copied without scaling.
            const int prefix = (start_layer < m) ? offsets[start_layer] : total_cols;
            if (prefix > 0) {
                std::memcpy(dst, src, static_cast<std::size_t>(prefix) * sizeof(float));
            }
            for (int l = start_layer; l < m; ++l) {
                const int begin = offsets[l];
                const int end = (l + 1 < m) ? offsets[l + 1] : total_cols;
                const float s = scales[l];
                const float* __restrict ps = src + begin;
                float* __restrict pd = dst + begin;
                const int len = end - begin;
#pragma omp simd
                for (int i = 0; i < len; ++i) {
                    pd[i] = s * ps[i];
                }
            }
        }

        // Templated timing implementation.
        // ProfileTiming controls whether timing bookkeeping exists in the hot kernel
        // (no runtime `if (timing)` branches inside the loops).
        template <bool ProfileTiming, typename ParentT>
        STLQ_ALWAYS_INLINE void ScanOneQueryLookupLinkagedScaledTablesImpl(const int cid,
                                                                           const int m,
                                                                           const int n_real,
                                                                           const int n_virt,
                                                                           const std::uint32_t* __restrict real_ids,
                                                                           const ParentT* __restrict parent_1based,
                                                                           const std::uint32_t* __restrict
                                                                           depth_offsets,
                                                                           const std::uint8_t* __restrict
                                                                           codes_small_bytes,
                                                                           const std::uint8_t* __restrict
                                                                           code0_one_bytes,
                                                                           const std::uint8_t* __restrict
                                                                           virt_codes_small_bytes,
                                                                           const float* __restrict r_norm2,
                                                                           const float* __restrict scales_root,
                                                                           const float* __restrict scales_linkage,
                                                                           const std::int8_t* __restrict q_layer_major,
                                                                           const float* __restrict xCq_root0_col,
                                                                           const int* __restrict offsets_root_small,
                                                                           const float* __restrict xCq_root_small_col,
                                                                           int root_small_total_cols,
                                                                           const int* __restrict offsets_one,
                                                                           const float* __restrict xCq_one_col,
                                                                           int one_total_cols,
                                                                           TopKHeap* __restrict topk,
                                                                           ScanScratch* __restrict scratch,
                                                                           ScanKernelTiming* __restrict timing) {
            const int nc = n_real + n_virt;
            const int real_base = n_virt;
            if (scratch->dot_all.size() < static_cast<std::size_t>(nc)) {
                scratch->dot_all.resize(static_cast<std::size_t>(nc));
            }
            float* __restrict dot_all = scratch->dot_all.data();

            constexpr int kMaxM = 64;
            const std::int8_t* qptr_stack[kMaxM] = {};
            const float* oneptr_stack[kMaxM] = {};
            const std::int8_t** __restrict qptr = qptr_stack;
            const float** __restrict oneptr = oneptr_stack;
#ifndef NDEBUG
    if (m > kMaxM) {
        throw std::runtime_error("ScanOneQueryLookupLinkagedScaledTables: m exceeds kMaxM.");
    }
#endif

            double t0 = 0.0;
            if constexpr (ProfileTiming) {
                t0 = omp_get_wtime();
            }
            if (scratch->root_small_scaled.size() < static_cast<std::size_t>(root_small_total_cols)) {
                scratch->root_small_scaled.resize(static_cast<std::size_t>(root_small_total_cols));
            }
            if (scratch->one_scaled.size() < static_cast<std::size_t>(one_total_cols)) {
                scratch->one_scaled.resize(static_cast<std::size_t>(one_total_cols));
            }
            CopyScalePackedTable(scratch->root_small_scaled.data(),
                                 xCq_root_small_col,
                                 offsets_root_small,
                                 root_small_total_cols,
                                 scales_root,
                                 m,
                                 /*start_layer=*/1);
            CopyScalePackedTable(scratch->one_scaled.data(),
                                 xCq_one_col,
                                 offsets_one,
                                 one_total_cols,
                                 scales_linkage,
                                 m,
                                 /*start_layer=*/0);
            if constexpr (ProfileTiming) {
                timing->t_scale_tables += omp_get_wtime() - t0;
            }

            const float root0_scaled = scales_root[0] * xCq_root0_col[cid];
            const float* __restrict root_small = scratch->root_small_scaled.data();
            const float* __restrict one = scratch->one_scaled.data();
            const std::ptrdiff_t small_layers = static_cast<std::ptrdiff_t>(m - 1);

            const int n_root_real = static_cast<int>(depth_offsets[1]);
            for (int l = 0; l < m; ++l) {
                qptr[l] = q_layer_major + static_cast<std::ptrdiff_t>(l) * nc;
                oneptr[l] = one + offsets_one[l];
            }
            const std::ptrdiff_t stride_codes = small_layers;
            const std::int8_t* __restrict q0 = qptr[0];

            // Roots (virt + real depth-0): layer-major accumulation.
            if constexpr (ProfileTiming) {
                t0 = omp_get_wtime();
            }
            const int n_root_real_offset = real_base + n_root_real;
#pragma omp simd
            for (int v = 0; v < n_virt; ++v) {
                dot_all[v] = root0_scaled * static_cast<float>(q0[v]);
            }
#pragma omp simd
            for (int pos = real_base; pos < n_root_real_offset; ++pos) {
                dot_all[pos] = root0_scaled * static_cast<float>(q0[pos]);
            }
            if constexpr (ProfileTiming) {
                timing->t_roots_init += omp_get_wtime() - t0;
            }

            if constexpr (ProfileTiming) {
                t0 = omp_get_wtime();
            }
#if defined(__GNUC__)
#pragma GCC unroll 16
#elif defined(_MSC_VER)
    #pragma loop( unroll(16) )
#endif
            for (int l = 1; l < m; ++l) {
                const std::int8_t* __restrict ql = qptr[l];
                const float* __restrict table = root_small + offsets_root_small[l];
                for (int v = 0; v < n_virt; ++v) {
                    const std::uint8_t* __restrict row =
                        virt_codes_small_bytes + static_cast<std::ptrdiff_t>(v) * stride_codes;
                    const std::uint8_t code = row[l - 1];
                    dot_all[v] += static_cast<float>(ql[v]) * table[static_cast<int>(code)];
                }
                for (int pos = 0; pos < n_root_real; ++pos) {
                    const int local = real_base + pos;
                    const std::uint8_t* __restrict row =
                        codes_small_bytes + static_cast<std::ptrdiff_t>(pos) * stride_codes;
                    const std::uint8_t code = row[l - 1];
                    dot_all[local] += static_cast<float>(ql[local]) * table[static_cast<int>(code)];
                }
            }
            if constexpr (ProfileTiming) {
                timing->t_roots_layers += omp_get_wtime() - t0;
            }

            // Linkage nodes: per-pos accumulation (parent contribution fused).
            if constexpr (ProfileTiming) {
                t0 = omp_get_wtime();
            }
            const float* __restrict one0 = oneptr[0];
            for (int pos = n_root_real; pos < n_real; ++pos) {
                const int local = real_base + pos;
#ifndef NDEBUG
        const int p1 = static_cast<int>(parent_1based[pos]);
        if (p1 <= 0 || p1 > nc) {
            throw std::runtime_error("ScanOneQueryLookupLinkagedScaledTables: invalid parent for linkage node.");
        }
        const int p = p1 - 1;
        if (p >= local) {
            throw std::runtime_error("ScanOneQueryLookupLinkagedScaledTables: parent not earlier than child.");
        }
#else
                const int p = static_cast<int>(parent_1based[pos]) - 1;
#endif
                float dot = 0.0f;
                const std::uint8_t code0 = ReadCode0One(code0_one_bytes, pos);
                dot += static_cast<float>(q0[local]) * one0[static_cast<int>(code0)];
                const std::uint8_t* __restrict row =
                    codes_small_bytes + static_cast<std::ptrdiff_t>(pos) * stride_codes;
#if defined(__GNUC__)
#pragma GCC unroll 16
#elif defined(_MSC_VER)
    #pragma loop( unroll(16) )
#endif
                for (int l = 1; l < m; ++l) {
                    const std::int8_t* __restrict ql = qptr[l];
                    const float* __restrict one_l = oneptr[l];
                    const std::uint8_t code = row[l - 1];
                    dot += static_cast<float>(ql[local]) * one_l[static_cast<int>(code)];
                }
                dot_all[local] = dot + dot_all[p];
            }
            if constexpr (ProfileTiming) {
                timing->t_linkage += omp_get_wtime() - t0;
            }

            // Dist + heap updates.
            double t_push0 = 0.0;
            if constexpr (ProfileTiming) {
                t_push0 = omp_get_wtime();
            }
            if (scratch->dist_tmp.size() < static_cast<std::size_t>(n_real)) {
                scratch->dist_tmp.resize(static_cast<std::size_t>(n_real));
            }
            float* __restrict dtmp = scratch->dist_tmp.data();
            double t1 = 0.0;
            if constexpr (ProfileTiming) {
                t1 = omp_get_wtime();
            }
#pragma omp simd
            for (int pos = 0; pos < n_real; ++pos) {
                dtmp[pos] = r_norm2[pos] - 2.0f * dot_all[real_base + pos];
            }
            if constexpr (ProfileTiming) {
                timing->t_push_dist += omp_get_wtime() - t1;
                t1 = omp_get_wtime();
            }
            const int heap_k = topk->k;
            if (static_cast<int>(topk->heap.size()) < heap_k) {
                for (int pos = 0; pos < n_real; ++pos) {
                    topk->Push(dtmp[pos], static_cast<std::int32_t>(real_ids[pos]));
                }
            }
            else {
                if (!topk->heapified) topk->Heapify();
                Candidate worst = topk->heap[0];
                for (int pos = 0; pos < n_real; ++pos) {
                    const float dist = dtmp[pos];
                    const auto id = static_cast<std::int32_t>(real_ids[pos]);
                    if (dist > worst.dist || (dist == worst.dist && id >= worst.id)) continue;
                    topk->PushAssumeBetter(dist, id);
                    worst = topk->heap[0];
                }
            }
            if constexpr (ProfileTiming) {
                timing->t_push_heap += omp_get_wtime() - t1;
                timing->t_push += omp_get_wtime() - t_push0;
            }
        }

        template <bool ProfileTiming, typename ParentT>
        STLQ_ALWAYS_INLINE void ScanOneQueryLookupLinkagedScaledTablesLutImpl(const int cid,
                                                                              const int m,
                                                                              const int n_real,
                                                                              const int n_virt,
                                                                              const std::uint32_t* __restrict real_ids,
                                                                              const ParentT* __restrict parent_1based,
                                                                              const std::uint32_t* __restrict
                                                                              depth_offsets,
                                                                              const std::uint8_t* __restrict
                                                                              codes_small_bytes,
                                                                              const std::uint8_t* __restrict
                                                                              code0_one_bytes,
                                                                              const std::uint8_t* __restrict
                                                                              virt_codes_small_bytes,
                                                                              const std::uint8_t* __restrict
                                                                              r_norm2_codes,
                                                                              const float* __restrict r_norm2_centers,
                                                                              const float* __restrict scales_root,
                                                                              const float* __restrict scales_linkage,
                                                                              const std::int8_t* __restrict
                                                                              q_layer_major,
                                                                              const float* __restrict xCq_root0_col,
                                                                              const int* __restrict offsets_root_small,
                                                                              const float* __restrict
                                                                              xCq_root_small_col,
                                                                              int root_small_total_cols,
                                                                              const int* __restrict offsets_one,
                                                                              const float* __restrict xCq_one_col,
                                                                              int one_total_cols,
                                                                              TopKHeap* __restrict topk,
                                                                              ScanScratch* __restrict scratch,
                                                                              ScanKernelTiming* __restrict timing) {
            const int nc = n_real + n_virt;
            const int real_base = n_virt;
            if (scratch->dot_all.size() < static_cast<std::size_t>(nc)) {
                scratch->dot_all.resize(static_cast<std::size_t>(nc));
            }
            float* __restrict dot_all = scratch->dot_all.data();

            constexpr int kMaxM = 64;
            const std::int8_t* qptr_stack[kMaxM] = {};
            const float* oneptr_stack[kMaxM] = {};
            const std::int8_t** __restrict qptr = qptr_stack;
            const float** __restrict oneptr = oneptr_stack;
#ifndef NDEBUG
    if (m > kMaxM) {
        throw std::runtime_error("ScanOneQueryLookupLinkagedScaledTablesLut: m exceeds kMaxM.");
    }
#endif

            double t0 = 0.0;
            if constexpr (ProfileTiming) {
                t0 = omp_get_wtime();
            }
            if (scratch->root_small_scaled.size() < static_cast<std::size_t>(root_small_total_cols)) {
                scratch->root_small_scaled.resize(static_cast<std::size_t>(root_small_total_cols));
            }
            if (scratch->one_scaled.size() < static_cast<std::size_t>(one_total_cols)) {
                scratch->one_scaled.resize(static_cast<std::size_t>(one_total_cols));
            }
            CopyScalePackedTable(scratch->root_small_scaled.data(),
                                 xCq_root_small_col,
                                 offsets_root_small,
                                 root_small_total_cols,
                                 scales_root,
                                 m,
                                 /*start_layer=*/1);
            CopyScalePackedTable(scratch->one_scaled.data(),
                                 xCq_one_col,
                                 offsets_one,
                                 one_total_cols,
                                 scales_linkage,
                                 m,
                                 /*start_layer=*/0);
            if constexpr (ProfileTiming) {
                timing->t_scale_tables += omp_get_wtime() - t0;
            }

            const float root0_scaled = scales_root[0] * xCq_root0_col[cid];
            const float* __restrict root_small = scratch->root_small_scaled.data();
            const float* __restrict one = scratch->one_scaled.data();
            const std::ptrdiff_t small_layers = static_cast<std::ptrdiff_t>(m - 1);

            const int n_root_real = static_cast<int>(depth_offsets[1]);
            for (int l = 0; l < m; ++l) {
                qptr[l] = q_layer_major + static_cast<std::ptrdiff_t>(l) * nc;
                oneptr[l] = one + offsets_one[l];
            }
            const std::ptrdiff_t stride_codes = small_layers;
            const std::int8_t* __restrict q0 = qptr[0];

            if constexpr (ProfileTiming) {
                t0 = omp_get_wtime();
            }
            const int n_root_real_offset = real_base + n_root_real;
#pragma omp simd
            for (int v = 0; v < n_virt; ++v) {
                dot_all[v] = root0_scaled * static_cast<float>(q0[v]);
            }
#pragma omp simd
            for (int pos = real_base; pos < n_root_real_offset; ++pos) {
                dot_all[pos] = root0_scaled * static_cast<float>(q0[pos]);
            }
            if constexpr (ProfileTiming) {
                timing->t_roots_init += omp_get_wtime() - t0;
            }

            if constexpr (ProfileTiming) {
                t0 = omp_get_wtime();
            }
#if defined(__GNUC__)
#pragma GCC unroll 16
#elif defined(_MSC_VER)
    #pragma loop( unroll(16) )
#endif
            for (int l = 1; l < m; ++l) {
                const std::int8_t* __restrict ql = qptr[l];
                const float* __restrict table = root_small + offsets_root_small[l];
                for (int v = 0; v < n_virt; ++v) {
                    const std::uint8_t* __restrict row =
                        virt_codes_small_bytes + static_cast<std::ptrdiff_t>(v) * stride_codes;
                    const std::uint8_t code = row[l - 1];
                    dot_all[v] += static_cast<float>(ql[v]) * table[static_cast<int>(code)];
                }
                for (int pos = 0; pos < n_root_real; ++pos) {
                    const int local = real_base + pos;
                    const std::uint8_t* __restrict row =
                        codes_small_bytes + static_cast<std::ptrdiff_t>(pos) * stride_codes;
                    const std::uint8_t code = row[l - 1];
                    dot_all[local] += static_cast<float>(ql[local]) * table[static_cast<int>(code)];
                }
            }
            if constexpr (ProfileTiming) {
                timing->t_roots_layers += omp_get_wtime() - t0;
            }

            if constexpr (ProfileTiming) {
                t0 = omp_get_wtime();
            }
            const float* __restrict one0 = oneptr[0];
            for (int pos = n_root_real; pos < n_real; ++pos) {
                const int local = real_base + pos;
#ifndef NDEBUG
        const int p1 = static_cast<int>(parent_1based[pos]);
        if (p1 <= 0 || p1 > nc) {
            throw std::runtime_error("ScanOneQueryLookupLinkagedScaledTablesLut: invalid parent for linkage node.");
        }
        const int p = p1 - 1;
        if (p >= local) {
            throw std::runtime_error("ScanOneQueryLookupLinkagedScaledTablesLut: parent not earlier than child.");
        }
#else
                const int p = static_cast<int>(parent_1based[pos]) - 1;
#endif
                float dot = 0.0f;
                const std::uint8_t code0 = ReadCode0One(code0_one_bytes, pos);
                dot += static_cast<float>(q0[local]) * one0[static_cast<int>(code0)];
                const std::uint8_t* __restrict row =
                    codes_small_bytes + static_cast<std::ptrdiff_t>(pos) * stride_codes;
#if defined(__GNUC__)
#pragma GCC unroll 16
#elif defined(_MSC_VER)
    #pragma loop( unroll(16) )
#endif
                for (int l = 1; l < m; ++l) {
                    const std::int8_t* __restrict ql = qptr[l];
                    const float* __restrict one_l = oneptr[l];
                    const std::uint8_t code = row[l - 1];
                    dot += static_cast<float>(ql[local]) * one_l[static_cast<int>(code)];
                }
                dot_all[local] = dot + dot_all[p];
            }
            if constexpr (ProfileTiming) {
                timing->t_linkage += omp_get_wtime() - t0;
            }

            double t_push0 = 0.0;
            if constexpr (ProfileTiming) {
                t_push0 = omp_get_wtime();
            }
            if (scratch->dist_tmp.size() < static_cast<std::size_t>(n_real)) {
                scratch->dist_tmp.resize(static_cast<std::size_t>(n_real));
            }
            float* __restrict dtmp = scratch->dist_tmp.data();
            double t1 = 0.0;
            if constexpr (ProfileTiming) {
                t1 = omp_get_wtime();
            }
#pragma omp simd
            for (int pos = 0; pos < n_real; ++pos) {
                dtmp[pos] = r_norm2_centers[static_cast<int>(r_norm2_codes[pos])] - 2.0f * dot_all[real_base + pos];
            }
            if constexpr (ProfileTiming) {
                timing->t_push_dist += omp_get_wtime() - t1;
                t1 = omp_get_wtime();
            }
            const int heap_k = topk->k;
            if (static_cast<int>(topk->heap.size()) < heap_k) {
                for (int pos = 0; pos < n_real; ++pos) {
                    topk->Push(dtmp[pos], static_cast<std::int32_t>(real_ids[pos]));
                }
            }
            else {
                if (!topk->heapified) topk->Heapify();
                Candidate worst = topk->heap[0];
                for (int pos = 0; pos < n_real; ++pos) {
                    const float dist = dtmp[pos];
                    const auto id = static_cast<std::int32_t>(real_ids[pos]);
                    if (dist > worst.dist || (dist == worst.dist && id >= worst.id)) continue;
                    topk->PushAssumeBetter(dist, id);
                    worst = topk->heap[0];
                }
            }
            if constexpr (ProfileTiming) {
                timing->t_push_heap += omp_get_wtime() - t1;
                timing->t_push += omp_get_wtime() - t_push0;
            }
        }

        STLQ_ALWAYS_INLINE void ScanOneQueryLookupLinkagedScaledTables(const eval::ClusterView& cv,
                                                                       const float* __restrict xCq_root0_col,
                                                                       const int* __restrict offsets_root_small,
                                                                       const float* __restrict xCq_root_small_col,
                                                                       int root_small_total_cols,
                                                                       const int* __restrict offsets_one,
                                                                       const float* __restrict xCq_one_col,
                                                                       int one_total_cols,
                                                                       TopKHeap* __restrict topk,
                                                                       ScanScratch* __restrict scratch,
                                                                       ScanKernelTiming* __restrict timing) {
            if (cv.r_norm2_lut_u8) {
                if (cv.parent_is_u16) {
                    if (timing) {
                        ScanOneQueryLookupLinkagedScaledTablesLutImpl<true>(
                            cv.cid, cv.m, cv.n_real, cv.n_virt,
                            cv.real_ids, cv.parent_1based_u16, cv.depth_offsets,
                            cv.codes_small_bytes, cv.code0_one_bytes, cv.virt_codes_small_bytes,
                            cv.r_norm2_lut_u8, cv.r_norm2_lut_centers,
                            cv.scales_root, cv.scales_linkage, cv.q_layer_major,
                            xCq_root0_col, offsets_root_small, xCq_root_small_col, root_small_total_cols,
                            offsets_one, xCq_one_col, one_total_cols,
                            topk, scratch, timing);
                    }
                    else {
                        ScanOneQueryLookupLinkagedScaledTablesLutImpl<false>(
                            cv.cid, cv.m, cv.n_real, cv.n_virt,
                            cv.real_ids, cv.parent_1based_u16, cv.depth_offsets,
                            cv.codes_small_bytes, cv.code0_one_bytes, cv.virt_codes_small_bytes,
                            cv.r_norm2_lut_u8, cv.r_norm2_lut_centers,
                            cv.scales_root, cv.scales_linkage, cv.q_layer_major,
                            xCq_root0_col, offsets_root_small, xCq_root_small_col, root_small_total_cols,
                            offsets_one, xCq_one_col, one_total_cols,
                            topk, scratch, nullptr);
                    }
                }
                else {
                    if (timing) {
                        ScanOneQueryLookupLinkagedScaledTablesLutImpl<true>(
                            cv.cid, cv.m, cv.n_real, cv.n_virt,
                            cv.real_ids, cv.parent_1based, cv.depth_offsets,
                            cv.codes_small_bytes, cv.code0_one_bytes, cv.virt_codes_small_bytes,
                            cv.r_norm2_lut_u8, cv.r_norm2_lut_centers,
                            cv.scales_root, cv.scales_linkage, cv.q_layer_major,
                            xCq_root0_col, offsets_root_small, xCq_root_small_col, root_small_total_cols,
                            offsets_one, xCq_one_col, one_total_cols,
                            topk, scratch, timing);
                    }
                    else {
                        ScanOneQueryLookupLinkagedScaledTablesLutImpl<false>(
                            cv.cid, cv.m, cv.n_real, cv.n_virt,
                            cv.real_ids, cv.parent_1based, cv.depth_offsets,
                            cv.codes_small_bytes, cv.code0_one_bytes, cv.virt_codes_small_bytes,
                            cv.r_norm2_lut_u8, cv.r_norm2_lut_centers,
                            cv.scales_root, cv.scales_linkage, cv.q_layer_major,
                            xCq_root0_col, offsets_root_small, xCq_root_small_col, root_small_total_cols,
                            offsets_one, xCq_one_col, one_total_cols,
                            topk, scratch, nullptr);
                    }
                }
                return;
            }
            if (cv.parent_is_u16) {
                if (timing) {
                    ScanOneQueryLookupLinkagedScaledTablesImpl<true>(
                        cv.cid, cv.m, cv.n_real, cv.n_virt,
                        cv.real_ids, cv.parent_1based_u16, cv.depth_offsets,
                        cv.codes_small_bytes, cv.code0_one_bytes, cv.virt_codes_small_bytes,
                        cv.r_norm2, cv.scales_root, cv.scales_linkage, cv.q_layer_major,
                        xCq_root0_col, offsets_root_small, xCq_root_small_col, root_small_total_cols,
                        offsets_one, xCq_one_col, one_total_cols,
                        topk, scratch, timing);
                }
                else {
                    ScanOneQueryLookupLinkagedScaledTablesImpl<false>(
                        cv.cid, cv.m, cv.n_real, cv.n_virt,
                        cv.real_ids, cv.parent_1based_u16, cv.depth_offsets,
                        cv.codes_small_bytes, cv.code0_one_bytes, cv.virt_codes_small_bytes,
                        cv.r_norm2, cv.scales_root, cv.scales_linkage, cv.q_layer_major,
                        xCq_root0_col, offsets_root_small, xCq_root_small_col, root_small_total_cols,
                        offsets_one, xCq_one_col, one_total_cols,
                        topk, scratch, nullptr);
                }
            }
            else {
                if (timing) {
                    ScanOneQueryLookupLinkagedScaledTablesImpl<true>(
                        cv.cid, cv.m, cv.n_real, cv.n_virt,
                        cv.real_ids, cv.parent_1based, cv.depth_offsets,
                        cv.codes_small_bytes, cv.code0_one_bytes, cv.virt_codes_small_bytes,
                        cv.r_norm2, cv.scales_root, cv.scales_linkage, cv.q_layer_major,
                        xCq_root0_col, offsets_root_small, xCq_root_small_col, root_small_total_cols,
                        offsets_one, xCq_one_col, one_total_cols,
                        topk, scratch, timing);
                }
                else {
                    ScanOneQueryLookupLinkagedScaledTablesImpl<false>(
                        cv.cid, cv.m, cv.n_real, cv.n_virt,
                        cv.real_ids, cv.parent_1based, cv.depth_offsets,
                        cv.codes_small_bytes, cv.code0_one_bytes, cv.virt_codes_small_bytes,
                        cv.r_norm2, cv.scales_root, cv.scales_linkage, cv.q_layer_major,
                        xCq_root0_col, offsets_root_small, xCq_root_small_col, root_small_total_cols,
                        offsets_one, xCq_one_col, one_total_cols,
                        topk, scratch, nullptr);
                }
            }
        }

        template <typename ParentT>
        STLQ_ALWAYS_INLINE void ScanOneQueryLookupLinkagedFloatLutImpl(const int cid,
                                                                       const int m,
                                                                       const int m_codes,
                                                                       const int n_real,
                                                                       const int n_virt,
                                                                       const std::uint32_t* __restrict real_ids,
                                                                       const ParentT* __restrict parent_1based,
                                                                       const std::uint32_t* __restrict depth_offsets,
                                                                       const std::uint8_t* __restrict codes_small_bytes,
                                                                       const std::uint8_t* __restrict code0_one_bytes,
                                                                       const std::uint8_t* __restrict
                                                                       virt_codes_small_bytes,
                                                                       const std::uint8_t* __restrict r_norm2_codes,
                                                                       const float* __restrict r_norm2_centers,
                                                                       const float* __restrict a0,
                                                                       const float* __restrict coeffs_small,
                                                                       const float* __restrict virt_a0,
                                                                       const float* __restrict virt_coeffs_small,
                                                                       const float* __restrict xCq_root0_col,
                                                                       const int* __restrict offsets_root_small,
                                                                       const float* __restrict xCq_root_small_col,
                                                                       const int* __restrict offsets_one,
                                                                       const float* __restrict xCq_one_col,
                                                                       TopKHeap* __restrict topk,
                                                                       ScanScratch* __restrict scratch) {
            const int nc = n_real + n_virt;
            const int real_base = n_virt;
            if (scratch->dot_all.size() < static_cast<std::size_t>(nc)) {
                scratch->dot_all.resize(static_cast<std::size_t>(nc));
            }
            float* __restrict dot_all = scratch->dot_all.data();
            const std::ptrdiff_t small_layers = static_cast<std::ptrdiff_t>(m_codes);

            const int n_root_real = static_cast<int>(depth_offsets[1]);

            constexpr int kMaxM = 64;
            const float* rootptr_stack[kMaxM];
            const float* oneptr_stack[kMaxM];
            const float** __restrict rootptr = rootptr_stack;
            const float** __restrict oneptr = oneptr_stack;
#ifndef NDEBUG
    if (m > kMaxM) {
        throw std::runtime_error("ScanOneQueryLookupLinkagedFloatLut: m exceeds kMaxM.");
    }
#endif
            rootptr[0] = nullptr;
            oneptr[0] = xCq_one_col + offsets_one[0];
            for (int l = 1; l < m; ++l) {
                rootptr[l] = xCq_root_small_col + offsets_root_small[l];
                oneptr[l] = xCq_one_col + offsets_one[l];
            }

            const float root0 = xCq_root0_col[cid];
            const std::ptrdiff_t stride_codes = small_layers;

#pragma omp simd
            for (int v = 0; v < n_virt; ++v) {
                dot_all[v] = virt_a0[v] * root0;
            }
#pragma omp simd
            for (int pos = 0; pos < n_root_real; ++pos) {
                const int local = real_base + pos;
                dot_all[local] = a0[pos] * root0;
            }
#if defined(__GNUC__)
#pragma GCC unroll 16
#elif defined(_MSC_VER)
    #pragma loop( unroll(16) )
#endif
            for (int l = 1; l < m; ++l) {
                const float* __restrict table = rootptr[l];
                for (int v = 0; v < n_virt; ++v) {
                    const std::uint8_t* __restrict row =
                        virt_codes_small_bytes + static_cast<std::ptrdiff_t>(v) * stride_codes;
                    const float* __restrict arow =
                        virt_coeffs_small + static_cast<std::ptrdiff_t>(v) * small_layers;
                    const std::uint8_t code = row[l - 1];
                    dot_all[v] += arow[l - 1] * table[static_cast<int>(code)];
                }
                for (int pos = 0; pos < n_root_real; ++pos) {
                    const int local = real_base + pos;
                    const std::uint8_t* __restrict row =
                        codes_small_bytes + static_cast<std::ptrdiff_t>(pos) * stride_codes;
                    const float* __restrict arow =
                        coeffs_small + static_cast<std::ptrdiff_t>(pos) * small_layers;
                    const std::uint8_t code = row[l - 1];
                    dot_all[local] += arow[l - 1] * table[static_cast<int>(code)];
                }
            }

            const float* __restrict one0 = oneptr[0];
            for (int pos = n_root_real; pos < n_real; ++pos) {
                const int local = real_base + pos;
#ifndef NDEBUG
        const int p1 = static_cast<int>(parent_1based[pos]);
        if (p1 <= 0 || p1 > nc) {
            throw std::runtime_error("ScanOneQueryLookupLinkagedFloatLut: invalid parent for linkage node.");
        }
        const int p = p1 - 1;
        if (p >= local) {
            throw std::runtime_error("ScanOneQueryLookupLinkagedFloatLut: parent not earlier than child.");
        }
#else
                const int p = static_cast<int>(parent_1based[pos]) - 1;
#endif
                float dot = 0.0f;
                const std::uint8_t code0 = ReadCode0One(code0_one_bytes, pos);
                dot += a0[pos] * one0[static_cast<int>(code0)];

                const std::uint8_t* __restrict row =
                    codes_small_bytes + static_cast<std::ptrdiff_t>(pos) * stride_codes;
                const float* __restrict arow =
                    coeffs_small + static_cast<std::ptrdiff_t>(pos) * small_layers;
#if defined(__GNUC__)
#pragma GCC unroll 16
#elif defined(_MSC_VER)
    #pragma loop( unroll(16) )
#endif
                for (int l = 1; l < m; ++l) {
                    const float* __restrict one_l = oneptr[l];
                    const std::uint8_t code = row[l - 1];
                    dot += arow[l - 1] * one_l[static_cast<int>(code)];
                }
                dot_all[local] = dot + dot_all[p];
            }

            if (scratch->dist_tmp.size() < static_cast<std::size_t>(n_real)) {
                scratch->dist_tmp.resize(static_cast<std::size_t>(n_real));
            }
            float* __restrict dtmp = scratch->dist_tmp.data();
#pragma omp simd
            for (int pos = 0; pos < n_real; ++pos) {
                dtmp[pos] = r_norm2_centers[static_cast<int>(r_norm2_codes[pos])] - 2.0f * dot_all[real_base + pos];
            }
            const int heap_k = topk->k;
            if (static_cast<int>(topk->heap.size()) < heap_k) {
                for (int pos = 0; pos < n_real; ++pos) {
                    topk->Push(dtmp[pos], static_cast<std::int32_t>(real_ids[pos]));
                }
            }
            else {
                if (!topk->heapified) topk->Heapify();
                Candidate worst = topk->heap[0];
                for (int pos = 0; pos < n_real; ++pos) {
                    const float dist = dtmp[pos];
                    const auto id = static_cast<std::int32_t>(real_ids[pos]);
                    if (dist > worst.dist || (dist == worst.dist && id >= worst.id)) continue;
                    topk->PushAssumeBetter(dist, id);
                    worst = topk->heap[0];
                }
            }
        }

        template <typename ParentT>
        STLQ_ALWAYS_INLINE void ScanOneQueryLookupLinkagedFloatImpl(const int cid,
                                                                    const int m,
                                                                    const int m_codes,
                                                                    const int n_real,
                                                                    const int n_virt,
                                                                    const std::uint32_t* __restrict real_ids,
                                                                    const ParentT* __restrict parent_1based,
                                                                    const std::uint32_t* __restrict depth_offsets,
                                                                    const std::uint8_t* __restrict codes_small_bytes,
                                                                    const std::uint8_t* __restrict code0_one_bytes,
                                                                    const std::uint8_t* __restrict
                                                                    virt_codes_small_bytes,
                                                                    const float* __restrict r_norm2,
                                                                    const float* __restrict a0,
                                                                    const float* __restrict coeffs_small,
                                                                    const float* __restrict virt_a0,
                                                                    const float* __restrict virt_coeffs_small,
                                                                    const float* __restrict xCq_root0_col,
                                                                    const int* __restrict offsets_root_small,
                                                                    const float* __restrict xCq_root_small_col,
                                                                    const int* __restrict offsets_one,
                                                                    const float* __restrict xCq_one_col,
                                                                    TopKHeap* __restrict topk,
                                                                    ScanScratch* __restrict scratch) {
            const int nc = n_real + n_virt;
            const int real_base = n_virt;
            if (scratch->dot_all.size() < static_cast<std::size_t>(nc)) {
                scratch->dot_all.resize(static_cast<std::size_t>(nc));
            }
            float* __restrict dot_all = scratch->dot_all.data();
            const std::ptrdiff_t small_layers = static_cast<std::ptrdiff_t>(m_codes);

            const int n_root_real = static_cast<int>(depth_offsets[1]);

            constexpr int kMaxM = 64;
            const float* rootptr_stack[kMaxM];
            const float* oneptr_stack[kMaxM];
            const float** __restrict rootptr = rootptr_stack;
            const float** __restrict oneptr = oneptr_stack;
#ifndef NDEBUG
	if (m > kMaxM) {
	    throw std::runtime_error("ScanOneQueryLookupLinkagedFloat: m exceeds kMaxM.");
	}
#endif
            rootptr[0] = nullptr;
            oneptr[0] = xCq_one_col + offsets_one[0];
            for (int l = 1; l < m; ++l) {
                rootptr[l] = xCq_root_small_col + offsets_root_small[l];
                oneptr[l] = xCq_one_col + offsets_one[l];
            }

            const float root0 = xCq_root0_col[cid];
            const std::ptrdiff_t stride_codes = small_layers;

            // 1) Roots (virtual + real depth-0): accumulate by layer (l outer, pos inner).
#pragma omp simd
            for (int v = 0; v < n_virt; ++v) {
                dot_all[v] = virt_a0[v] * root0;
            }
#pragma omp simd
            for (int pos = 0; pos < n_root_real; ++pos) {
                const int local = real_base + pos;
                dot_all[local] = a0[pos] * root0;
            }
#if defined(__GNUC__)
#pragma GCC unroll 16
#elif defined(_MSC_VER)
    #pragma loop( unroll(16) )
#endif
            for (int l = 1; l < m; ++l) {
                const float* __restrict table = rootptr[l];
                for (int v = 0; v < n_virt; ++v) {
                    const std::uint8_t* __restrict row =
                        virt_codes_small_bytes + static_cast<std::ptrdiff_t>(v) * stride_codes;
                    const float* __restrict arow =
                        virt_coeffs_small + static_cast<std::ptrdiff_t>(v) * small_layers;
                    const std::uint8_t code = row[l - 1];
                    dot_all[v] += arow[l - 1] * table[static_cast<int>(code)];
                }
                for (int pos = 0; pos < n_root_real; ++pos) {
                    const int local = real_base + pos;
                    const std::uint8_t* __restrict row =
                        codes_small_bytes + static_cast<std::ptrdiff_t>(pos) * stride_codes;
                    const float* __restrict arow =
                        coeffs_small + static_cast<std::ptrdiff_t>(pos) * small_layers;
                    const std::uint8_t code = row[l - 1];
                    dot_all[local] += arow[l - 1] * table[static_cast<int>(code)];
                }
            }

            // 2) Linkage nodes: per-pos accumulation (parent contribution fused).
            const float* __restrict one0 = oneptr[0];
            for (int pos = n_root_real; pos < n_real; ++pos) {
                const int local = real_base + pos;
#ifndef NDEBUG
	    const int p1 = static_cast<int>(parent_1based[pos]);
	    if (p1 <= 0 || p1 > nc) {
	        throw std::runtime_error("ScanOneQueryLookupLinkagedFloat: invalid parent for linkage node.");
	    }
	    const int p = p1 - 1;
	    if (p >= local) {
	        throw std::runtime_error("ScanOneQueryLookupLinkagedFloat: parent not earlier than child.");
	    }
#else
                const int p = static_cast<int>(parent_1based[pos]) - 1;
#endif
                float dot = 0.0f;
                const std::uint8_t code0 = ReadCode0One(code0_one_bytes, pos);
                dot += a0[pos] * one0[static_cast<int>(code0)];

                const std::uint8_t* __restrict row =
                    codes_small_bytes + static_cast<std::ptrdiff_t>(pos) * stride_codes;
                const float* __restrict arow =
                    coeffs_small + static_cast<std::ptrdiff_t>(pos) * small_layers;
#if defined(__GNUC__)
#pragma GCC unroll 16
#elif defined(_MSC_VER)
    #pragma loop( unroll(16) )
#endif
                for (int l = 1; l < m; ++l) {
                    const float* __restrict one_l = oneptr[l];
                    const std::uint8_t code = row[l - 1];
                    dot += arow[l - 1] * one_l[static_cast<int>(code)];
                }
                dot_all[local] = dot + dot_all[p];
            }

            if (scratch->dist_tmp.size() < static_cast<std::size_t>(n_real)) {
                scratch->dist_tmp.resize(static_cast<std::size_t>(n_real));
            }
            float* __restrict dtmp = scratch->dist_tmp.data();
#pragma omp simd
            for (int pos = 0; pos < n_real; ++pos) {
                dtmp[pos] = r_norm2[pos] - 2.0f * dot_all[real_base + pos];
            }
            const int heap_k = topk->k;
            if (static_cast<int>(topk->heap.size()) < heap_k) {
                for (int pos = 0; pos < n_real; ++pos) {
                    topk->Push(dtmp[pos], static_cast<std::int32_t>(real_ids[pos]));
                }
            }
            else {
                if (!topk->heapified) topk->Heapify();
                Candidate worst = topk->heap[0];
                for (int pos = 0; pos < n_real; ++pos) {
                    const float dist = dtmp[pos];
                    const auto id = static_cast<std::int32_t>(real_ids[pos]);
                    if (dist > worst.dist || (dist == worst.dist && id >= worst.id)) continue;
                    topk->PushAssumeBetter(dist, id);
                    worst = topk->heap[0];
                }
            }
        }

        STLQ_ALWAYS_INLINE void ScanOneQueryLookupLinkagedFloat(const eval::ClusterView& cv,
                                                                const float* __restrict xCq_root0_col,
                                                                const int* __restrict offsets_root_small,
                                                                const float* __restrict xCq_root_small_col,
                                                                const int* __restrict offsets_one,
                                                                const float* __restrict xCq_one_col,
                                                                TopKHeap* __restrict topk,
                                                                ScanScratch* __restrict scratch) {
            if (cv.r_norm2_lut_u8) {
                if (cv.parent_is_u16) {
                    ScanOneQueryLookupLinkagedFloatLutImpl(
                        cv.cid, cv.m, cv.m_codes, cv.n_real, cv.n_virt,
                        cv.real_ids, cv.parent_1based_u16, cv.depth_offsets,
                        cv.codes_small_bytes, cv.code0_one_bytes, cv.virt_codes_small_bytes,
                        cv.r_norm2_lut_u8, cv.r_norm2_lut_centers,
                        cv.a0, cv.coeffs_small, cv.virt_a0, cv.virt_coeffs_small,
                        xCq_root0_col, offsets_root_small, xCq_root_small_col, offsets_one,
                        xCq_one_col, topk, scratch);
                }
                else {
                    ScanOneQueryLookupLinkagedFloatLutImpl(
                        cv.cid, cv.m, cv.m_codes, cv.n_real, cv.n_virt,
                        cv.real_ids, cv.parent_1based, cv.depth_offsets,
                        cv.codes_small_bytes, cv.code0_one_bytes, cv.virt_codes_small_bytes,
                        cv.r_norm2_lut_u8, cv.r_norm2_lut_centers,
                        cv.a0, cv.coeffs_small, cv.virt_a0, cv.virt_coeffs_small,
                        xCq_root0_col, offsets_root_small, xCq_root_small_col, offsets_one,
                        xCq_one_col, topk, scratch);
                }
                return;
            }
            if (cv.parent_is_u16) {
                ScanOneQueryLookupLinkagedFloatImpl(
                    cv.cid, cv.m, cv.m_codes, cv.n_real, cv.n_virt,
                    cv.real_ids, cv.parent_1based_u16, cv.depth_offsets,
                    cv.codes_small_bytes, cv.code0_one_bytes, cv.virt_codes_small_bytes,
                    cv.r_norm2, cv.a0, cv.coeffs_small, cv.virt_a0, cv.virt_coeffs_small,
                    xCq_root0_col, offsets_root_small, xCq_root_small_col, offsets_one,
                    xCq_one_col, topk, scratch);
            }
            else {
                ScanOneQueryLookupLinkagedFloatImpl(
                    cv.cid, cv.m, cv.m_codes, cv.n_real, cv.n_virt,
                    cv.real_ids, cv.parent_1based, cv.depth_offsets,
                    cv.codes_small_bytes, cv.code0_one_bytes, cv.virt_codes_small_bytes,
                    cv.r_norm2, cv.a0, cv.coeffs_small, cv.virt_a0, cv.virt_coeffs_small,
                    xCq_root0_col, offsets_root_small, xCq_root_small_col, offsets_one,
                    xCq_one_col, topk, scratch);
            }
        }

        static void RunScanTopK_FloatCoeffCpu(const STLQueryTables& qt,
                                              const std::vector<int>& offsets_root_small,
                                              const CodebookMeta& meta_one,
                                              const std::vector<int>& q_cids_len,
                                              const std::vector<int>& q_cids_flat,
                                              const std::vector<int>& cid_to_active,
                                              const std::vector<eval::ClusterView>& active_views,
                                              int q0,
                                              int qlen,
                                              int nprobe_cap,
                                              std::vector<TopKHeap>* topk,
                                              RecallResult* out,
                                              ScanTopKProfileSums* sums) {
            if (!topk || !out) return;

            const bool want_profile = (sums && sums->enabled);
            const double scan_t0 = (want_profile && sums->total_scan_kernel) ? omp_get_wtime() : 0.0;

#pragma omp parallel default(none) shared(topk, q_cids_len, q_cids_flat, cid_to_active, active_views, qt, offsets_root_small, meta_one) firstprivate(qlen, nprobe_cap)
            {
                ScanScratch scratch;
#pragma omp for schedule(static)
                for (int qi = 0; qi < qlen; ++qi) {
                    TopKHeap& heap = (*topk)[static_cast<std::size_t>(qi)];
                    const int nprobe = q_cids_len[static_cast<std::size_t>(qi)];
                    for (int t = 0; t < nprobe; ++t) {
                        const int cid =
                            q_cids_flat[static_cast<std::size_t>(qi) * static_cast<std::size_t>(nprobe_cap) +
                                static_cast<std::size_t>(t)];
                        const int idx = cid_to_active[static_cast<std::size_t>(cid)];
                        if (idx < 0) continue;
                        const auto& cv = active_views[static_cast<std::size_t>(idx)];
                        ScanOneQueryLookupLinkagedFloat(
                            cv,
                            qt.xCq_root0.Col(qi),
                            offsets_root_small.data(), qt.xCq_root_small.Col(qi),
                            meta_one.offsets.data(), qt.xCq_one.Col(qi),
                            &heap,
                            &scratch);
                    }
                }
            }

            if (want_profile && sums->total_scan_kernel) {
                *sums->total_scan_kernel += omp_get_wtime() - scan_t0;
            }

            const double finalize_t0 = (want_profile && sums->total_topk_finalize) ? omp_get_wtime() : 0.0;

            for (int qi = 0; qi < qlen; ++qi) {
                (*topk)[static_cast<std::size_t>(qi)].Finalize(out->dists.Col(q0 + qi), out->indices.Col(q0 + qi));
            }

            if (want_profile && sums->total_topk_finalize) {
                *sums->total_topk_finalize += omp_get_wtime() - finalize_t0;
            }
        }

        static void RunScanTopK_Int8CoeffCpuAllTasks(const STLQueryTables& qt,
                                                     const std::vector<int>& offsets_root_small,
                                                     const CodebookMeta& meta_one,
                                                     int root_small_total_cols,
                                                     int one_total_cols,
                                                     ScanCoeffFn scan_coeff_fn,
                                                     const std::vector<int>& q_cids_len,
                                                     const std::vector<int>& q_cids_flat,
                                                     const std::vector<int>& cid_to_active,
                                                     const std::vector<eval::ClusterView>& active_views,
                                                     int q0,
                                                     int qlen,
                                                     int nprobe_cap,
                                                     std::vector<TopKHeap>* topk,
                                                     RecallResult* out,
                                                     ScanTopKProfileSums* sums) {
            if (!topk || !out) return;
            const bool want_profile = (sums && sums->enabled);

            // Production keeps each query's finalization with its scan owner.  That avoids a
            // second parallel region/barrier, but scan and finalize then overlap across workers
            // and have no separately additive wall-time intervals.  Profiling deliberately uses
            // two globally separated phases so both numbers are measured outside their OpenMP
            // regions.  The production path below remains unchanged when profiling is disabled.
            if (want_profile) {
                const double scan_kernel_t0 = omp_get_wtime();
#pragma omp parallel default(none) shared(topk, out, q_cids_len, q_cids_flat, cid_to_active, active_views, qt, offsets_root_small, meta_one, scan_coeff_fn, sums) firstprivate(q0, qlen, nprobe_cap, root_small_total_cols, one_total_cols, want_profile)
                {
                    ScanScratch scratch;
                    ScanKernelTiming timing_local{};
#pragma omp for schedule(static)
                    for (int qi = 0; qi < qlen; ++qi) {
                        TopKHeap& heap = (*topk)[static_cast<std::size_t>(qi)];
                        const int nprobe = q_cids_len[static_cast<std::size_t>(qi)];
                        for (int t = 0; t < nprobe; ++t) {
                            const int cid =
                                q_cids_flat[static_cast<std::size_t>(qi) * static_cast<std::size_t>(nprobe_cap) +
                                    static_cast<std::size_t>(t)];
                            const int idx = cid_to_active[static_cast<std::size_t>(cid)];
                            if (idx < 0) continue;
                            const auto& cv = active_views[static_cast<std::size_t>(idx)];
                            scan_coeff_fn(cv,
                                          qt.xCq_root0.Col(qi),
                                          offsets_root_small.data(), qt.xCq_root_small.Col(qi),
                                          root_small_total_cols,
                                          meta_one.offsets.data(), qt.xCq_one.Col(qi), one_total_cols,
                                          &heap,
                                          &scratch,
                                          &timing_local);
                        }
                    }

                    if (sums->total_scan_scale_tables) {
#pragma omp atomic
                        *sums->total_scan_scale_tables += timing_local.t_scale_tables;
                    }
                    if (sums->total_scan_roots_init) {
#pragma omp atomic
                        *sums->total_scan_roots_init += timing_local.t_roots_init;
                    }
                    if (sums->total_scan_roots_layers) {
#pragma omp atomic
                        *sums->total_scan_roots_layers += timing_local.t_roots_layers;
                    }
                    if (sums->total_scan_linkage) {
#pragma omp atomic
                        *sums->total_scan_linkage += timing_local.t_linkage;
                    }
                    if (sums->total_scan_push) {
#pragma omp atomic
                        *sums->total_scan_push += timing_local.t_push;
                    }
                    if (sums->total_scan_push_dist) {
#pragma omp atomic
                        *sums->total_scan_push_dist += timing_local.t_push_dist;
                    }
                    if (sums->total_scan_push_heap) {
#pragma omp atomic
                        *sums->total_scan_push_heap += timing_local.t_push_heap;
                    }
                }
                if (sums->total_scan_kernel) {
                    *sums->total_scan_kernel += omp_get_wtime() - scan_kernel_t0;
                }

                const double finalize_t0 = omp_get_wtime();
#pragma omp parallel default(none) shared(topk, out, sums) firstprivate(q0, qlen)
                {
                    double finalize_worker_local = 0.0;
#pragma omp for schedule(static)
                    for (int qi = 0; qi < qlen; ++qi) {
                        const double query_finalize_t0 = omp_get_wtime();
                        (*topk)[static_cast<std::size_t>(qi)].Finalize(
                            out->dists.Col(q0 + qi),
                            out->indices.Col(q0 + qi));
                        finalize_worker_local += omp_get_wtime() - query_finalize_t0;
                    }
                    if (sums->total_topk_finalize_worker) {
#pragma omp atomic
                        *sums->total_topk_finalize_worker += finalize_worker_local;
                    }
                }
                if (sums->total_topk_finalize) {
                    *sums->total_topk_finalize += omp_get_wtime() - finalize_t0;
                }
                return;
            }

#pragma omp parallel default(none) shared(topk, out, q_cids_len, q_cids_flat, cid_to_active, active_views, qt, offsets_root_small, meta_one, scan_coeff_fn) firstprivate(q0, qlen, nprobe_cap, root_small_total_cols, one_total_cols)
            {
                ScanScratch scratch;
#pragma omp for schedule(static)
                for (int qi = 0; qi < qlen; ++qi) {
                    TopKHeap& heap = (*topk)[static_cast<std::size_t>(qi)];
                    const int nprobe = q_cids_len[static_cast<std::size_t>(qi)];
                    for (int t = 0; t < nprobe; ++t) {
                        const int cid =
                            q_cids_flat[static_cast<std::size_t>(qi) * static_cast<std::size_t>(nprobe_cap) +
                                static_cast<std::size_t>(t)];
                        const int idx = cid_to_active[static_cast<std::size_t>(cid)];
                        if (idx < 0) continue;
                        const auto& cv = active_views[static_cast<std::size_t>(idx)];
                        scan_coeff_fn(cv,
                                      qt.xCq_root0.Col(qi),
                                      offsets_root_small.data(), qt.xCq_root_small.Col(qi), root_small_total_cols,
                                      meta_one.offsets.data(), qt.xCq_one.Col(qi), one_total_cols,
                                      &heap,
                                      &scratch,
                                      nullptr);
                    }
                    // Each query heap is private to this OpenMP iteration. Keep ownership through
                    // final sorting in the production path instead of introducing another global
                    // phase boundary.
                    heap.Finalize(
                        out->dists.Col(q0 + qi),
                        out->indices.Col(q0 + qi));
                }
            }
        }

        // CPU fallback for GPU scan: compute local TopK for a subset of (query,cluster) tasks.
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
        static double RunScanTopK_Int8CoeffCpuTasksLocalTopK(const STLQueryTables& qt,
                                                             const std::vector<int>& offsets_root_small,
                                                             const CodebookMeta& meta_one,
                                                             int root_small_total_cols,
                                                             int one_total_cols,
                                                             ScanCoeffFn scan_coeff_fn,
                                                             const std::vector<eval::ClusterView>& active_views,
                                                             const std::vector<std::pair<int, int>>& tasks,
                                                             // (task, active_idx)
                                                             int nprobe_cap,
                                                             int k,
                                                             std::vector<float>* out_dists,
                                                             std::vector<std::uint32_t>* out_ids,
                                                             bool want_profile) {
            if (!out_dists || !out_ids) return 0.0;
            if (tasks.empty()) {
                out_dists->clear();
                out_ids->clear();
                return 0.0;
            }
            const double t0 = want_profile ? omp_get_wtime() : 0.0;
            out_dists->resize(tasks.size() * static_cast<std::size_t>(k));
            out_ids->resize(tasks.size() * static_cast<std::size_t>(k));

            const int ntasks = static_cast<int>(tasks.size());
#pragma omp parallel default(none) shared(tasks, active_views, qt, offsets_root_small, meta_one, scan_coeff_fn, out_dists, out_ids) firstprivate(ntasks, nprobe_cap, k, root_small_total_cols, one_total_cols)
            {
                ScanScratch scratch;
                std::vector<float> tmp_d(static_cast<std::size_t>(k));
                std::vector<int> tmp_i(static_cast<std::size_t>(k));
#pragma omp for schedule(static)
                for (int ii = 0; ii < ntasks; ++ii) {
                    const auto& fb = tasks[static_cast<std::size_t>(ii)];
                    const int task = fb.first;
                    const int idx = fb.second;
                    const int qi = task / nprobe_cap;
                    const auto& cv = active_views[static_cast<std::size_t>(idx)];
                    TopKHeap heap_local(k);
                    scan_coeff_fn(cv,
                                  qt.xCq_root0.Col(qi),
                                  offsets_root_small.data(), qt.xCq_root_small.Col(qi), root_small_total_cols,
                                  meta_one.offsets.data(), qt.xCq_one.Col(qi), one_total_cols,
                                  &heap_local,
                                  &scratch,
                                  nullptr);
                    heap_local.Finalize(tmp_d.data(), tmp_i.data());
                    const std::size_t base = static_cast<std::size_t>(ii) * static_cast<std::size_t>(k);
                    for (int j = 0; j < k; ++j) {
                        (*out_dists)[base + static_cast<std::size_t>(j)] = tmp_d[static_cast<std::size_t>(j)];
                        const int gi = tmp_i[static_cast<std::size_t>(j)];
                        (*out_ids)[base + static_cast<std::size_t>(j)] =
                            (gi < 0) ? 0xFFFFFFFFu : static_cast<std::uint32_t>(gi);
                    }
                }
            }

            return want_profile ? (omp_get_wtime() - t0) : 0.0;
        }

        static void BuildGpuScanTasks(const std::vector<int>& q_cids_len,
                                      const std::vector<int>& q_cids_flat,
                                      const std::vector<int>& cid_to_active,
                                      const std::vector<eval::ClusterView>& active_views,
                                      int qlen,
                                      int nprobe_cap,
                                      int effective_max_nc,
                                      std::vector<int>* task_cluster_idx,
                                      std::vector<std::pair<int, int>>* cpu_fallback) {
            if (!task_cluster_idx || !cpu_fallback) return;
            task_cluster_idx->assign(static_cast<std::size_t>(qlen) * static_cast<std::size_t>(nprobe_cap), -1);
            cpu_fallback->clear();
            cpu_fallback->reserve(task_cluster_idx->size());
            for (int qi = 0; qi < qlen; ++qi) {
                const int nprobe = q_cids_len[static_cast<std::size_t>(qi)];
                for (int t = 0; t < nprobe; ++t) {
                    const int cid =
                        q_cids_flat[static_cast<std::size_t>(qi) * static_cast<std::size_t>(nprobe_cap) +
                            static_cast<std::size_t>(t)];
                    const int idx = cid_to_active[static_cast<std::size_t>(cid)];
                    if (idx < 0) continue;
                    const auto& cv = active_views[static_cast<std::size_t>(idx)];
                    const int task = qi * nprobe_cap + t;
                    if (cv.nc <= effective_max_nc) {
                        (*task_cluster_idx)[static_cast<std::size_t>(task)] = idx;
                    }
                    else {
                        cpu_fallback->emplace_back(task, idx);
                    }
                }
            }
        }
#endif

        static void RunScanTopK_Int8Coeff(const Config& cfg,
                                          const io::LinkageListReader& linkage_list,
                                          const STLQueryTables& qt,
                                          int m,
                                          int m_codes,
                                          const std::vector<int>& offsets_root_small,
                                          const CodebookMeta& meta_one,
                                          int root_small_total_cols,
                                          int one_total_cols,
                                          ScanCoeffFn scan_coeff_fn,
                                          const std::vector<int>& q_cids_len,
                                          const std::vector<int>& q_cids_flat,
                                          const std::vector<int>& cid_to_active,
                                          const std::vector<eval::ClusterView>& active_views,
                                          int q0,
                                          int qlen,
                                          int nprobe_cap,
                                          int k,
                                          std::vector<TopKHeap>* topk,
                                          RecallResult* out,
                                          ScanTopKProfileSums* sums) {
            (void)cfg;
            (void)linkage_list;
            (void)m;
            (void)m_codes;
            (void)k;

#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
            const bool want_profile = (sums && sums->enabled);
            const bool want_gpu_scan = cfg.eval.linkage_gpu_scan_enable && cfg.runtime.use_cuda &&
                !eval::IsDiskNorm2ModeLut(cfg.eval.disk_norm2_mode);
            if (want_gpu_scan) {
                // IMPORTANT: For the user-facing "scan kernel time (wall)" we must use a true wall clock
                // measurement. Summing internal breakdown fields can undercount (e.g., stream sync or
                // scheduling overhead) and makes int8 path look unrealistically faster than float.
                const double gpu_scan_wall_t0 = want_profile ? omp_get_wtime() : 0.0;

                const int device_max_nc = stlq::eval::cuda::DiskLinkageGpuScanMaxNcSupported();
                const int effective_max_nc =
                    (device_max_nc > 0)
                        ? std::min(cfg.eval.linkage_gpu_scan_max_nc, device_max_nc)
                        : cfg.eval.linkage_gpu_scan_max_nc;

                std::vector<int> task_cluster_idx;
                std::vector<std::pair<int, int>> cpu_fallback;
                BuildGpuScanTasks(q_cids_len, q_cids_flat, cid_to_active, active_views,
                                  qlen, nprobe_cap, effective_max_nc,
                                  &task_cluster_idx, &cpu_fallback);

                std::vector<float> task_dists;
                std::vector<std::uint32_t> task_ids;
                stlq::eval::cuda::DiskLinkageGpuScanStats gpu_stats{};
                stlq::eval::cuda::DiskLinkageGpuScanStats* gpu_stats_ptr = want_profile ? &gpu_stats : nullptr;
                std::string gpu_err;

                const bool ok = stlq::eval::cuda::ScanDiskLinkageLookupTopK(
                    qt, m, m_codes,
                    /*nlist=*/linkage_list.nlist(),
                    root_small_total_cols,
                    one_total_cols,
                    offsets_root_small.data(),
                    meta_one.offsets.data(),
                    active_views,
                    task_cluster_idx,
                    qlen,
                    nprobe_cap,
                    k,
                    effective_max_nc,
                    cfg.eval.linkage_gpu_scan_cache_mb,
                    cfg.eval.linkage_gpu_scan_tile256,
                    &task_dists,
                    &task_ids,
                    /*stats=*/gpu_stats_ptr,
                    &gpu_err);

                const double gpu_scan_wall_sec = want_profile ? (omp_get_wtime() - gpu_scan_wall_t0) : 0.0;

                if (!ok) {
                    if (sums && sums->gpu_scan_failed_any && sums->gpu_scan_first_err) {
                        if (!(*sums->gpu_scan_failed_any)) {
                            *sums->gpu_scan_failed_any = true;
                            *sums->gpu_scan_first_err = gpu_err;
                        }
                    }
                    // GPU failed: fallback to CPU scan over all tasks.
                    RunScanTopK_Int8CoeffCpuAllTasks(
                        qt, offsets_root_small, meta_one,
                        root_small_total_cols, one_total_cols,
                        scan_coeff_fn,
                        q_cids_len, q_cids_flat, cid_to_active, active_views,
                        q0, qlen, nprobe_cap,
                        topk, out, sums);
                    return;
                }

                // GPU succeeded: optionally scan oversized clusters on CPU, then merge.
                std::vector<std::vector<std::size_t>> fallback_by_qi;
                std::vector<float> fb_dists;
                std::vector<std::uint32_t> fb_ids;
                double cpu_fallback_sec = 0.0;
                if (!cpu_fallback.empty()) {
                    fallback_by_qi.resize(static_cast<std::size_t>(qlen));
                    for (std::size_t ii = 0; ii < cpu_fallback.size(); ++ii) {
                        const int task = cpu_fallback[ii].first;
                        const int qi = task / nprobe_cap;
                        if (qi >= 0 && qi < qlen) {
                            fallback_by_qi[static_cast<std::size_t>(qi)].push_back(ii);
                        }
                    }
                    cpu_fallback_sec = RunScanTopK_Int8CoeffCpuTasksLocalTopK(
                        qt, offsets_root_small, meta_one,
                        root_small_total_cols, one_total_cols,
                        scan_coeff_fn,
                        active_views,
                        cpu_fallback,
                        nprobe_cap,
                        k,
                        &fb_dists,
                        &fb_ids,
                        want_profile);
                }

                const double merge_t0 = want_profile ? omp_get_wtime() : 0.0;
#pragma omp parallel default(none) shared(q_cids_len, task_ids, task_dists, cpu_fallback, fallback_by_qi, fb_ids, fb_dists, out) firstprivate(qlen, nprobe_cap, k, q0)
                {
#pragma omp for schedule(static)
                    for (int qi = 0; qi < qlen; ++qi) {
                        TopKHeap heap(k);
                        const int nprobe = q_cids_len[static_cast<std::size_t>(qi)];
                        for (int t = 0; t < nprobe; ++t) {
                            const std::size_t base =
                            (static_cast<std::size_t>(qi) * static_cast<std::size_t>(nprobe_cap) +
                                static_cast<std::size_t>(t)) * static_cast<std::size_t>(k);
                            for (int j = 0; j < k; ++j) {
                                const std::uint32_t id = task_ids[base + static_cast<std::size_t>(j)];
                                if (id == 0xFFFFFFFFu) continue;
                                heap.Push(task_dists[base + static_cast<std::size_t>(j)],
                                          static_cast<std::int32_t>(id));
                            }
                        }

                        if (!cpu_fallback.empty()) {
                            const auto& fb = fallback_by_qi[static_cast<std::size_t>(qi)];
                            for (std::size_t ii : fb) {
                                const std::size_t base = ii * static_cast<std::size_t>(k);
                                for (int j = 0; j < k; ++j) {
                                    const std::uint32_t id = fb_ids[base + static_cast<std::size_t>(j)];
                                    if (id == 0xFFFFFFFFu) continue;
                                    heap.Push(fb_dists[base + static_cast<std::size_t>(j)],
                                              static_cast<std::int32_t>(id));
                                }
                            }
                        }

                        heap.Finalize(out->dists.Col(q0 + qi), out->indices.Col(q0 + qi));
                    }
                }
                if (want_profile && sums && sums->total_topk_finalize) {
                    *sums->total_topk_finalize += omp_get_wtime() - merge_t0;
                }

                if (want_profile && sums) {
                    // Account scan time (GPU work + optional CPU fallback), excluding the merge/finalize pass.
                    // Use wall time for correctness; keep breakdown fields for diagnosis.
                    if (sums->total_scan_kernel) {
                        *sums->total_scan_kernel += gpu_scan_wall_sec + cpu_fallback_sec;
                    }
                    if (sums->total_gpu_scan_pack_h2d)
                        *sums->total_gpu_scan_pack_h2d += gpu_stats.
                            clusters_pack_h2d_sec;
                    if (sums->total_gpu_scan_host_pack) *sums->total_gpu_scan_host_pack += gpu_stats.host_pack_cpu_sec;
                    if (sums->total_gpu_scan_alloc) *sums->total_gpu_scan_alloc += gpu_stats.alloc_sec;
                    if (sums->total_gpu_scan_tables_h2d)
                        *sums->total_gpu_scan_tables_h2d += gpu_stats.
                            query_tables_h2d_sec;
                    if (sums->total_gpu_scan_kernel) *sums->total_gpu_scan_kernel += gpu_stats.kernel_sec;
                    if (sums->total_gpu_scan_k_roots) *sums->total_gpu_scan_k_roots += gpu_stats.kernel_roots_sec;
                    if (sums->total_gpu_scan_k_depth) *sums->total_gpu_scan_k_depth += gpu_stats.kernel_depth_sec;
                    if (sums->total_gpu_scan_k_topk) *sums->total_gpu_scan_k_topk += gpu_stats.kernel_topk_sec;
                    if (sums->total_gpu_scan_out_d2h) *sums->total_gpu_scan_out_d2h += gpu_stats.out_d2h_sec;
                    if (sums->total_gpu_scan_stream_sync)
                        *sums->total_gpu_scan_stream_sync += gpu_stats.
                            stream_sync_sec;
                    if (sums->total_gpu_scan_cpu_fallback) *sums->total_gpu_scan_cpu_fallback += cpu_fallback_sec;
                    if (sums->total_gpu_scan_tasks_total) *sums->total_gpu_scan_tasks_total += gpu_stats.tasks_total;
                    if (sums->total_gpu_scan_tasks_gpu) *sums->total_gpu_scan_tasks_gpu += gpu_stats.tasks_gpu;
                    if (sums->total_gpu_scan_clusters_total) {
                        *sums->total_gpu_scan_clusters_total = std::max(*sums->total_gpu_scan_clusters_total,
                                                                        gpu_stats.clusters_total);
                    }
                    if (sums->total_gpu_scan_clusters_gpu) {
                        *sums->total_gpu_scan_clusters_gpu = std::max(*sums->total_gpu_scan_clusters_gpu,
                                                                      gpu_stats.clusters_gpu);
                    }
                    if (sums->total_gpu_scan_kernel_max_nc) {
                        *sums->total_gpu_scan_kernel_max_nc = std::max(*sums->total_gpu_scan_kernel_max_nc,
                                                                       gpu_stats.kernel_max_nc);
                    }
                    if (sums->total_gpu_scan_cache_calls) *sums->total_gpu_scan_cache_calls += 1;
                    if (sums->total_gpu_scan_cache_enabled_calls) {
                        *sums->total_gpu_scan_cache_enabled_calls += (gpu_stats.cache_enabled != 0) ? 1 : 0;
                    }
                    if (sums->total_gpu_scan_cache_slots) {
                        *sums->total_gpu_scan_cache_slots = std::max(*sums->total_gpu_scan_cache_slots,
                                                                     gpu_stats.cache_slots);
                    }
                    if (sums->total_gpu_scan_cache_hits) *sums->total_gpu_scan_cache_hits += gpu_stats.cache_hits;
                    if (sums->total_gpu_scan_cache_misses) *sums->total_gpu_scan_cache_misses += gpu_stats.cache_misses;
                    if (sums->total_gpu_scan_cache_upload_clusters) {
                        *sums->total_gpu_scan_cache_upload_clusters += gpu_stats.cache_upload_clusters;
                    }
                    if (sums->total_gpu_scan_cache_upload_bytes) {
                        *sums->total_gpu_scan_cache_upload_bytes += gpu_stats.cache_upload_bytes;
                    }
                    if (sums->total_gpu_scan_task_cluster_idx_bytes) {
                        *sums->total_gpu_scan_task_cluster_idx_bytes += gpu_stats.task_cluster_idx_bytes;
                    }
                    if (sums->total_gpu_scan_out_d2h_bytes) {
                        *sums->total_gpu_scan_out_d2h_bytes += gpu_stats.out_d2h_bytes;
                    }
                }
                return;
            }
#endif

            // CPU scan only.
            RunScanTopK_Int8CoeffCpuAllTasks(
                qt, offsets_root_small, meta_one,
                root_small_total_cols, one_total_cols,
                scan_coeff_fn,
                q_cids_len, q_cids_flat, cid_to_active, active_views,
                q0, qlen, nprobe_cap,
                topk, out, sums);
        }

        // Persistent state shared across repeat eval calls via DiskLinkageEvalSession::provider_cache.
        // Bundles the coeff codec reader, the norm provider, and the ClusterProvider so that
        // preloading and file-handle setup only happen once (on first call).
        struct ProviderBundle
        {
            // Must outlive provider (provider holds raw pointers to these).
            io::LinkageCoeffCodecReader coeff_reader;
            // LinkageNormProviderLookup stores const references to these; store by value here so
            // the references remain valid for the lifetime of the bundle (across repeat calls).
            CodebookMeta meta_root_small_stored;
            CodebookMeta meta_one_stored;
            std::unique_ptr<stlq::eval::IClusterNormProvider> norm_owned;
            stlq::eval::ClusterProvider provider;
            // CPU-only decode times from the initial PreloadAllClusters call (no I/O time).
            // CPU-only decode times from the initial PreloadAllClusters call.
            // Stored for diagnostics only; NOT used for timing computation (bench or lazy handles that).
            double preload_louds_cpu_sec = 0.0;
            double preload_coeff_cpu_sec = 0.0;
            double preload_total_wall_sec = 0.0;
            int preload_io_threads_used = 0;

            // Benchmark results: average OMP-parallel wall time for LOUDS / Huffman decode
            // on all preloaded cluster ids. Set once; reused as a constant.
            bool bench_done = false;
            int bench_n_clusters = 0; // number of cluster ids used for preload benchmarking
            double bench_louds_wall_avg_sec = 0.0;
            double bench_coeff_wall_avg_sec = 0.0;
            // Profiling breakdown from the benchmark (filled when ProfileTiming is true).
            double bench_profile_louds_deser_sec = 0.0;
            double bench_profile_louds_decode_parent_sec = 0.0;
            double bench_profile_louds_slice_sec = 0.0;
            // Lazy-mode persistent timing summary (first cold-load session result, reused by repeat
            // runs and archive output even when later calls hit warm caches and produce zero local prep).
            std::vector<int> lazy_loaded_cids;
            bool lazy_bench_done = false;
            double lazy_louds_cpu_wall_sec = 0.0;
            double lazy_coeff_cpu_wall_sec = 0.0;
            double lazy_louds_total_wall_sec = 0.0;
            double lazy_coeff_total_wall_sec = 0.0;
            eval::ClusterProvider::ParentStorageStats parent_storage_stats{};
            // Logged once per session on first bundle creation.
            bool load_mode_logged = false;
            bool parent_cache_logged = false;
            bool hier2_logged = false;
            bool hnsw_logged = false;

            // Eval-only IVF probe HNSW index (over unit-normalized root centroids).
            std::shared_ptr<ivf::IvfProbeHnswIndex> ivf_probe_hnsw;
            // Set after the first eval call that outputs LOUDS/Huffman timing; suppresses repeat output.
            bool louds_timing_reported = false;
        };

        static void LogParentCacheMemoryOnce(const std::shared_ptr<ProviderBundle>& bundle) {
            if (!bundle || bundle->parent_cache_logged) return;
            bundle->parent_storage_stats = bundle->provider.GetParentStorageStats();
            const double parent_cache_baseline_mb =
                static_cast<double>(bundle->parent_storage_stats.baseline_parent_u32_bytes) / (1024.0 * 1024.0);
            const double parent_cache_resident_mb =
                static_cast<double>(bundle->parent_storage_stats.resident_parent_bytes) / (1024.0 * 1024.0);
            LogInfo("Disk IVF linkage LOUDS parent cache memory: baseline_u32=" + std::to_string(
                    parent_cache_baseline_mb) +
                "MB resident=" + std::to_string(parent_cache_resident_mb) +
                "MB clusters_u16=" + std::to_string(bundle->parent_storage_stats.clusters_u16) +
                " clusters_u32=" + std::to_string(bundle->parent_storage_stats.clusters_u32));
            bundle->parent_cache_logged = true;
        }

        static bool PreparePreloadDecodeBenchmark(const Config& cfg,
                                                  const io::LinkageListReader& linkage_list,
                                                  std::shared_ptr<ProviderBundle>* bundle_io,
                                                  bool profile_timing,
                                                  std::string* err) {
            auto& bundle = *bundle_io;
            if (!bundle || bundle->bench_done || cfg.eval.linkage_preload_clusters_io_threads < 0) return true;
            const int nlist = linkage_list.nlist();
            if (nlist <= 0) {
                bundle->bench_done = true;
                return true;
            }

            std::vector<int> all_cids(static_cast<std::size_t>(nlist));
            for (int cid = 0; cid < nlist; ++cid) {
                all_cids[static_cast<std::size_t>(cid)] = cid;
            }

            bundle->bench_n_clusters = nlist;
            eval::ClusterProvider::BenchmarkResult bench_res{};
            std::string bench_err;
            const int bench_times = std::max(1, cfg.eval.linkage_louds_huffman_bench_times);
            if (!bundle->provider.BenchmarkDecodeWall(all_cids, bench_times,
                                                      /*request_profile_breakdown=*/profile_timing,
                                                      &bench_res, &bench_err)) {
                if (err) *err = "Linkage recall: BenchmarkDecodeWall failed: " + bench_err;
                return false;
            }
            bundle->bench_louds_wall_avg_sec = bench_res.louds_wall_avg_sec;
            bundle->bench_coeff_wall_avg_sec = bench_res.coeff_wall_avg_sec;
            bundle->bench_profile_louds_deser_sec = bench_res.profile_louds_deser_sec;
            bundle->bench_profile_louds_decode_parent_sec = bench_res.profile_louds_decode_parent_sec;
            bundle->bench_profile_louds_slice_sec = bench_res.profile_louds_slice_sec;
            bundle->bench_done = true;
            bundle->provider.ReleaseCoeffRawPayloads();
            return true;
        }

        static bool EnsureDiskEvalProviderBundle(const Config& cfg,
                                                 const io::LinkageListReader& linkage_list,
                                                 const ColMajorMatrix<float>& C_root0,
                                                 const CodebookMeta& meta_root_small,
                                                 const CodebookMeta& meta_one,
                                                 const std::string& norm2_filename,
                                                 bool use_coeff_codec,
                                                 bool profile_timing,
                                                 DiskLinkageEvalTiming* /*timing*/,
                                                 DiskLinkageEvalSession* session,
                                                 std::shared_ptr<ProviderBundle>* bundle_out,
                                                 std::string* err) {
            const bool use_norm2_lut = eval::IsDiskNorm2ModeLut(cfg.eval.disk_norm2_mode);
            const bool use_global_lut = eval::IsDiskNorm2ModeLutGlobal(cfg.eval.disk_norm2_mode);
            const bool is_lazy_mode = (cfg.eval.linkage_preload_clusters_io_threads < 0);
            const bool first_preload = !(session && session->provider_cache);
            std::shared_ptr<ProviderBundle> bundle;
            if (!first_preload) {
                bundle = std::static_pointer_cast<ProviderBundle>(session->provider_cache);
            }
            else {
                bundle = std::make_shared<ProviderBundle>();
                bundle->meta_root_small_stored = meta_root_small;
                bundle->meta_one_stored = meta_one;

                if (use_coeff_codec) {
                    std::string local_err;
                    if (!bundle->coeff_reader.Open(linkage_list.dir(), &local_err)) {
                        if (err)
                            *err = local_err.empty()
                                       ? "EvaluateRecallLinkageIvfFromDisk(lookup): missing coeff codec store."
                                       : local_err;
                        return false;
                    }
                    if (bundle->coeff_reader.meta().nlist != linkage_list.nlist() ||
                        bundle->coeff_reader.meta().m != cfg.model.m) {
                        if (err) *err = "EvaluateRecallLinkageIvfFromDisk(lookup): coeff codec meta mismatch.";
                        return false;
                    }
                }

                if (use_global_lut) {
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
                            LogWarn("Disk IVF linkage eval: rebuilding stale global norm2 LUT cache (" +
                                (lut_err.empty() ? std::string("invalid cache") : lut_err) + ").");
                        }
                        else {
                            LogInfo("Precomputing full global norm2 LUT cache ...");
                        }
                        if (!BuildAndStoreFullNorm2LutCache(cfg,
                                                            linkage_list,
                                                            C_root0,
                                                            meta_root_small,
                                                            meta_one,
                                                            use_coeff_codec,
                                                            err)) {
                            return false;
                        }
                    }
                }

                bundle->norm_owned = MakeNormProviderOwned(cfg,
                                                           use_coeff_codec,
                                                           C_root0,
                                                           bundle->meta_root_small_stored,
                                                           bundle->meta_one_stored,
                                                           profile_timing);
                eval::IClusterNormProvider* norm_ptr = bundle->norm_owned.get();
                if (!bundle->provider.Open(linkage_list,
                                           (use_coeff_codec ? &bundle->coeff_reader : nullptr),
                                           norm_ptr,
                                           use_coeff_codec,
                                           cfg.eval.linkage_parent_louds_enable,
                                           static_cast<std::uint32_t>(std::max(1, cfg.eval.parent_louds_select_stride)),
                                           static_cast<std::uint32_t>(std::max(
                                               1, cfg.eval.parent_louds_rank_words_per_super_log2)),
                                           cfg.eval.parent_louds_build_indices,
                                           cfg.eval.linkage_parent_adaptive_u16_cache,
                                           use_norm2_lut,
                                           eval::IsDiskNorm2ModeLutGlobal(cfg.eval.disk_norm2_mode),
                                           cfg.base.encode.hnorms,
                                           cfg.eval.disk_norm2_lut_kmeans_niter,
                                           profile_timing,
                                           err)) {
                    return false;
                }

                if (use_norm2_lut) {
                    std::string lut_err;
                    if (use_global_lut) {
                        if (is_lazy_mode) {
                            std::vector<float> centers;
                            if (!eval::LoadNorm2LutCenters(linkage_list, use_coeff_codec,
                                                           cfg.base.encode.hnorms,
                                                           cfg.eval.disk_norm2_lut_kmeans_niter,
                                                           cfg.eval.disk_norm2_mode,
                                                           cfg.eval.disk_norm2_lut_log_alpha,
                                                           cfg.eval.disk_norm2_lut_piecewise_p1,
                                                           cfg.eval.disk_norm2_lut_piecewise_p2,
                                                           cfg.eval.disk_norm2_lut_piecewise_count_weight,
                                                           cfg.eval.disk_norm2_lut_piecewise_range_weight,
                                                           &centers, &lut_err)) {
                                if (err)
                                    *err = "Disk IVF linkage eval: failed to load global norm2 LUT centers: " +
                                        lut_err;
                                return false;
                            }
                            bundle->provider.SetGlobalNorm2LutCenters(std::move(centers));
                            bundle->provider.SetNorm2DiskLazyEnabled(true);
                        }
                        else {
                            eval::Norm2Lut preloaded_lut;
                            const auto load_res =
                                eval::LoadNorm2LutCache(linkage_list, use_coeff_codec,
                                                        cfg.base.encode.hnorms,
                                                        cfg.eval.disk_norm2_lut_kmeans_niter,
                                                        cfg.eval.disk_norm2_mode,
                                                        cfg.eval.disk_norm2_lut_log_alpha,
                                                        cfg.eval.disk_norm2_lut_piecewise_p1,
                                                        cfg.eval.disk_norm2_lut_piecewise_p2,
                                                        cfg.eval.disk_norm2_lut_piecewise_count_weight,
                                                        cfg.eval.disk_norm2_lut_piecewise_range_weight,
                                                        &preloaded_lut, &lut_err);
                            if (load_res != eval::Norm2LutDiskLoadResult::kLoaded) {
                                if (err)
                                    *err = "Disk IVF linkage eval: failed to load global norm2 LUT cache: " +
                                        lut_err;
                                return false;
                            }
                            bundle->provider.PreloadGlobalNorm2LutCache(std::move(preloaded_lut));
                        }
                    }
                    else if (is_lazy_mode) {
                        const auto check_res =
                            eval::CheckNorm2LutClusterCache(linkage_list, use_coeff_codec,
                                                            cfg.base.encode.hnorms,
                                                            cfg.eval.disk_norm2_lut_kmeans_niter,
                                                            &lut_err);
                        if (check_res == eval::Norm2LutDiskLoadResult::kInvalid) {
                            LogWarn("Disk IVF linkage eval: ignoring stale norm2 LUT cache (" + lut_err + ").");
                        }
                        else if (check_res == eval::Norm2LutDiskLoadResult::kLoaded) {
                            bundle->provider.SetNorm2DiskLazyEnabled(true);
                        }
                    }
                    else {
                        std::unordered_map<int, eval::Norm2Lut> preloaded_lut;
                        const auto load_res =
                            eval::LoadNorm2LutClusterCache(linkage_list, use_coeff_codec,
                                                           cfg.base.encode.hnorms,
                                                           cfg.eval.disk_norm2_lut_kmeans_niter,
                                                           &preloaded_lut, &lut_err);
                        if (load_res == eval::Norm2LutDiskLoadResult::kInvalid) {
                            LogWarn("Disk IVF linkage eval: ignoring stale norm2 LUT cache (" + lut_err + ").");
                        }
                        else if (load_res == eval::Norm2LutDiskLoadResult::kLoaded) {
                            bundle->provider.PreloadNorm2LutCache(std::move(preloaded_lut));
                        }
                    }
                }
                else {
                    const auto norm2_path = std::filesystem::path(linkage_list.dir()) / norm2_filename;
                    std::error_code fec;
                    if (std::filesystem::exists(norm2_path, fec)) {
                        std::string cache_hash_reason;
                        if (!CheckFloatNorm2Cache(linkage_list, use_coeff_codec, norm2_path, &cache_hash_reason)) {
                            LogWarn("Disk IVF linkage eval: ignoring stale norm2 cache " +
                                norm2_path.filename().string() + " (" + cache_hash_reason + ").");
                        }
                        else if (is_lazy_mode) {
                            bundle->provider.SetNorm2DiskLazyEnabled(true);
                        }
                        else {
                            const auto& real_offs = linkage_list.real_offsets();
                            const int nlist = linkage_list.nlist();
                            const std::uint64_t total_real = linkage_list.total_real();
                            std::ifstream fin(norm2_path, std::ios::binary);
                            if (fin) {
                                std::vector<float> flat(static_cast<std::size_t>(total_real));
                                fin.read(reinterpret_cast<char*>(flat.data()),
                                         static_cast<std::streamsize>(flat.size() * sizeof(float)));
                                if (fin) {
                                    std::unordered_map<int, std::vector<float>> preloaded;
                                    for (int cid = 0; cid < nlist; ++cid) {
                                        const auto lo = real_offs[static_cast<std::size_t>(cid)];
                                        const auto hi = real_offs[static_cast<std::size_t>(cid + 1)];
                                        const auto nr = hi - lo;
                                        if (nr == 0) continue;
                                        std::vector<float> v(static_cast<std::size_t>(nr));
                                        std::memcpy(v.data(), &flat[static_cast<std::size_t>(lo)],
                                                    static_cast<std::size_t>(nr) * sizeof(float));
                                        preloaded.emplace(cid, std::move(v));
                                    }
                                    bundle->provider.PreloadNorm2Cache(std::move(preloaded));
                                }
                            }
                        }
                    }
                }

                if (cfg.eval.linkage_preload_clusters_io_threads >= 0) {
                    const int io_thr = cfg.eval.linkage_preload_clusters_io_threads;
                    const std::string thr_str = (io_thr == 0) ? "follow_omp" : std::to_string(io_thr);
                    const int omp_default = GetOmpDefaultThreads();
                    bundle->preload_io_threads_used = std::max(1, std::min(
                                                                   (io_thr == 0) ? std::max(1, omp_default) : io_thr,
                                                                   linkage_list.nlist()));
                    LogInfo("Linkage recall: preloading all cluster data (io_threads=" + thr_str +
                        ", actual_io_threads=" + std::to_string(bundle->preload_io_threads_used) + ") ...");
                    Timer preload_timer;
                    std::string preload_err;
                    eval::ClusterProvider::PreloadStats pstats;
                    if (!bundle->provider.PreloadAllClusters(io_thr, &pstats, &preload_err)) {
                        if (err) *err = "Linkage recall: PreloadAllClusters failed: " + preload_err;
                        return false;
                    }
                    bundle->preload_louds_cpu_sec = pstats.louds_cpu_decode_sec;
                    bundle->preload_coeff_cpu_sec = pstats.coeff_cpu_decode_sec;
                    bundle->preload_total_wall_sec = preload_timer.ElapsedSeconds();
                    LogInfo("Linkage recall: cluster preload done in " +
                        std::to_string(bundle->preload_total_wall_sec) + " s");
                }

                if (session) session->provider_cache = bundle;
            }

            if (!bundle->load_mode_logged && !cfg.eval.bench_quiet) {
                const int bench_times_cfg = std::max(1, cfg.eval.linkage_louds_huffman_bench_times);
                const std::string omp_decode_note = ", omp_decode_threads=" + std::to_string(OmpMaxThreads());
                if (is_lazy_mode) {
                    LogInfo(
                        "LOUDS/Huffman load mode = lazy (accumulated from GetCluster cache-misses" + omp_decode_note +
                        ")");
                }
                else {
                    LogInfo("LOUDS/Huffman load mode = preload (bench_times=" +
                        std::to_string(bench_times_cfg) + ", cluster_count=" +
                        std::to_string(linkage_list.nlist()) + ", preload_io_threads=" +
                        std::to_string(bundle->preload_io_threads_used) + omp_decode_note + ")");
                }
                LogInfo(std::string("LOUDS parent cache mode = ") +
                    (cfg.eval.linkage_parent_adaptive_u16_cache ? "adaptive_u16" : "baseline_u32"));
                bundle->load_mode_logged = true;
            }
            if (!is_lazy_mode && !bundle->parent_cache_logged && !cfg.eval.bench_quiet) {
                LogParentCacheMemoryOnce(bundle);
            }
            if (bundle_out) *bundle_out = bundle;
            return true;
        }
    } // namespace

    static bool EvalDiskLinkageRecallLookupImpl(const Config& cfg,
                                                const Dataset& query_dataset_inmem,
                                                const io::LinkageListReader& linkage_list,
                                                const TrainResult& train,
                                                bool use_coeff_codec,
                                                RecallResult* out,
                                                DiskLinkageEvalTiming* timing,
                                                DiskLinkageEvalSession* session,
                                                std::string* err) {
        const bool profile_timing = cfg.large.profile_timing;
        // Read qt_rotate_wall_sec from timing (caller sets it before calling; 0 if timing is null).
        const double qt_rotate_wall_sec = timing ? timing->qt_rotate_wall_sec : 0.0;

        if (!cfg.eval.linkage_use_ivf_disk) {
            if (err) *err = "EvaluateRecallLinkageIvfFromDisk(lookup): eval.linkage.use_ivf_disk is false.";
            return false;
        }

        if (!out) {
            return false;
        }
        if (query_dataset_inmem.gt.size() != static_cast<std::size_t>(query_dataset_inmem.Xq.cols)) {
            if (err) *err = "EvaluateRecallLinkageIvfFromDisk(lookup): invalid ground truth.";
            return false;
        }
        // linkage_list is the canonical source of list partitioning for disk linkage recall.

        const int nquery = query_dataset_inmem.Xq.cols;
        const int d = query_dataset_inmem.Xq.rows;
        const int m = cfg.model.m;
        const int m_codes = std::max(0, m - 1);
        if (d <= 0 || m <= 1 || m_codes != linkage_list.m_codes()) {
            if (err) *err = "EvaluateRecallLinkageIvfFromDisk(lookup): invalid dims / m_codes mismatch.";
            return false;
        }
        if (static_cast<int>(train.C_root.books.size()) != m ||
            static_cast<int>(train.C_one.books.size()) != m) {
            if (err) *err = "EvaluateRecallLinkageIvfFromDisk(lookup): codebook count mismatch.";
            return false;
        }

        const int k = std::max(1, std::min(cfg.dataset.k, static_cast<int>(linkage_list.total_real())));
        const int nprobe_cap = std::min(std::max(1, cfg.eval.linkage_nprobe), linkage_list.nlist());
        const int qblk = std::max(1, cfg.eval.linkage_query_block);

        out->indices = ColMajorMatrix<int>(k, nquery);
        out->dists = ColMajorMatrix<float>(k, nquery);

        // ---- C_root0 and codebook meta: computed each call (used for query tables and validation) ----
        const ColMajorMatrix<float>& C_root0 = train.C_root.books.front();
        if (C_root0.rows != d || C_root0.cols != linkage_list.nlist()) {
            if (err) *err = "EvaluateRecallLinkageIvfFromDisk(lookup): invalid C_root[0] dims.";
            return false;
        }

        std::vector<const ColMajorMatrix<float>*> root_small_books;
        root_small_books.reserve(static_cast<std::size_t>(m_codes));
        for (int l = 1; l < m; ++l) {
            root_small_books.push_back(&train.C_root.books[static_cast<std::size_t>(l)]);
        }
        const CodebookMeta meta_root_small = BuildCodebookMeta(root_small_books);
        const CodebookMeta meta_one = BuildCodebookMeta(GatherBooks(train.C_one));
        if (meta_one.m != m || meta_one.d != d ||
            meta_root_small.m != m_codes || meta_root_small.d != d ||
            meta_one.sizes.empty() || meta_root_small.sizes.empty()) {
            if (err) *err = "EvaluateRecallLinkageIvfFromDisk(lookup): invalid CodebookMeta.";
            return false;
        }

        std::vector<int> offsets_root_small(static_cast<std::size_t>(m), 0);
        for (int l = 1; l < m; ++l) {
            offsets_root_small[static_cast<std::size_t>(l)] =
                meta_root_small.offsets[static_cast<std::size_t>(l - 1)];
        }
        const int root_small_total_cols = meta_root_small.total_cols;
        const int one_total_cols = meta_one.total_cols;
        (void)root_small_total_cols;
        (void)one_total_cols;

        ScanCoeffFn scan_coeff_fn = nullptr;
        if (use_coeff_codec) {
            scan_coeff_fn = &ScanOneQueryLookupLinkagedScaledTables;
        }

        const std::string norm2_filename = use_coeff_codec ? "norm2_int8.f32" : "norm2_float.f32";
        const bool store_has_parent_louds = (linkage_list.meta().store_parent_louds != 0);

        const bool is_lazy_mode = (cfg.eval.linkage_preload_clusters_io_threads < 0);
        std::shared_ptr<ProviderBundle> bundle;
        if (!EnsureDiskEvalProviderBundle(
            cfg, linkage_list, C_root0, meta_root_small, meta_one, norm2_filename,
            use_coeff_codec, profile_timing, timing, session, &bundle, err)) {
            return false;
        }
        if (!PreparePreloadDecodeBenchmark(cfg, linkage_list, &bundle, profile_timing, err)) {
            return false;
        }

        eval::ClusterProvider& provider = bundle->provider;
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
        // Re-derive the CUDA norm provider pointer (used for per-batch GPU stats in the query loop).
        eval::cuda::LinkageNormProviderLookupCuda* norm_provider_cuda = nullptr;
        norm_provider_cuda = dynamic_cast<eval::cuda::LinkageNormProviderLookupCuda*>(bundle->norm_owned.get());
#endif

        double total_norm = 0.0;
        double total_parent_louds_decode = 0.0;
        double total_parent_louds_read = 0.0;
        double total_parent_louds_deser = 0.0;
        double total_parent_louds_decode_parent = 0.0;
        double total_parent_louds_slice = 0.0;
        [[maybe_unused]] double total_coarse_gemm = 0.0;
        [[maybe_unused]] double total_table_gemm = 0.0;
        double total_coeff_decode = 0.0; // lazy: CPU-only Huffman decode; preload: avg wall (no I/O)
        // Per-call lazy decode counters (reset each eval call; 0 in preload mode since caches are warm).
        int lazy_louds_decoded_count = 0;
        int lazy_coeff_decoded_count = 0;
        // Lazy-mode total wall = I/O + CPU (for log annotation; archived as timing->louds/huffman_wall_sec).
        double total_parent_louds_io_wall = 0.0; // sum(parent_louds_read_sec + parent_louds_decode_sec)
        double total_coeff_io_wall = 0.0; // sum(coeff_io_sec) — disk read portion only
        // bench_times: >=1 always; used in preload mode to run BenchmarkDecodeWall; ignored in lazy mode.
        const int bench_times = std::max(1, cfg.eval.linkage_louds_huffman_bench_times);
        [[maybe_unused]] double total_scan_kernel = 0.0;
        [[maybe_unused]] double total_topk_finalize = 0.0;
        [[maybe_unused]] double total_topk_finalize_worker = 0.0;
        [[maybe_unused]] double total_scan_scale_tables = 0.0;
        [[maybe_unused]] double total_scan_roots_init = 0.0;
        [[maybe_unused]] double total_scan_roots_layers = 0.0;
        [[maybe_unused]] double total_scan_linkage = 0.0;
        [[maybe_unused]] double total_scan_push = 0.0;
        [[maybe_unused]] double total_scan_push_dist = 0.0;
        [[maybe_unused]] double total_scan_push_heap = 0.0;
        [[maybe_unused]] double total_norm_cpu_c0dot = 0.0;
        [[maybe_unused]] double total_norm_gpu_h2d = 0.0;
        [[maybe_unused]] double total_norm_gpu_kernel = 0.0;
        [[maybe_unused]] double total_norm_gpu_d2h = 0.0;
        [[maybe_unused]] int total_norm_gpu_used = 0;
        [[maybe_unused]] int total_norm_gpu_fallback = 0;
        [[maybe_unused]] int total_norm_gpu_fallback_reason[5] = {0, 0, 0, 0, 0}; // idx: 0..4
        [[maybe_unused]] int norm_gpu_first_too_large_cid = -1;
        [[maybe_unused]] int norm_gpu_first_too_large_nreal = 0;
        [[maybe_unused]] int norm_gpu_first_too_large_nc = 0;
        [[maybe_unused]] double total_gpu_scan_pack_h2d = 0.0;
        [[maybe_unused]] double total_gpu_scan_host_pack = 0.0;
        [[maybe_unused]] double total_gpu_scan_alloc = 0.0;
        [[maybe_unused]] double total_gpu_scan_tables_h2d = 0.0;
        [[maybe_unused]] double total_gpu_scan_kernel = 0.0;
        [[maybe_unused]] double total_gpu_scan_k_roots = 0.0;
        [[maybe_unused]] double total_gpu_scan_k_depth = 0.0;
        [[maybe_unused]] double total_gpu_scan_k_topk = 0.0;
        [[maybe_unused]] double total_gpu_scan_out_d2h = 0.0;
        [[maybe_unused]] double total_gpu_scan_stream_sync = 0.0;
        [[maybe_unused]] double total_gpu_scan_cpu_fallback = 0.0;
        [[maybe_unused]] int total_gpu_scan_tasks_total = 0;
        [[maybe_unused]] int total_gpu_scan_tasks_gpu = 0;
        [[maybe_unused]] int total_gpu_scan_clusters_total = 0;
        [[maybe_unused]] int total_gpu_scan_clusters_gpu = 0;
        [[maybe_unused]] int total_gpu_scan_kernel_max_nc = 0;
        [[maybe_unused]] int total_gpu_scan_cache_slots = 0;
        [[maybe_unused]] int total_gpu_scan_cache_calls = 0;
        [[maybe_unused]] int total_gpu_scan_cache_enabled_calls = 0;
        [[maybe_unused]] int total_gpu_scan_cache_hits = 0;
        [[maybe_unused]] int total_gpu_scan_cache_misses = 0;
        [[maybe_unused]] int total_gpu_scan_cache_upload_clusters = 0;
        [[maybe_unused]] std::uint64_t total_gpu_scan_cache_upload_bytes = 0;
        [[maybe_unused]] std::uint64_t total_gpu_scan_task_cluster_idx_bytes = 0;
        [[maybe_unused]] std::uint64_t total_gpu_scan_out_d2h_bytes = 0;
        [[maybe_unused]] bool gpu_scan_failed_any = false;
        [[maybe_unused]] std::string gpu_scan_first_err;

        // "Core" wall timings (for QPS):
        // - qt_gemm_wall : pure GemmRaw time (coarse GEMM + table GEMMs); same definition for exact and hier2.
        // - probe_sel_wall: probe selection after GEMM:
        //     exact  => SelectTopClustersByScore (cheap linear scan, usually negligible)
        //     hier2  => SelectTopCoarseByScore + fine dot-product scan + partial_sort per query
        // - scan_topk_wall: scan kernel + topk finalize wall time.
        // core = qt_gemm_wall + probe_sel_wall + scan_topk_wall  (excludes setup/alloc/cluster-load).
        double total_core_qt_wall = 0.0;
        double total_core_qt_gemm_wall = 0.0;
        double total_core_qt_coarse_gemm_wall = 0.0;
        double total_core_qt_root_small_gemm_wall = 0.0;
        double total_core_qt_one_gemm_wall = 0.0;
        double total_core_probe_sel_wall = 0.0;
        double total_core_scan_topk_wall = 0.0;

        const int nlist = linkage_list.nlist();
        std::vector<std::uint8_t> active_flag(static_cast<std::size_t>(nlist), 0);
        std::vector<int> active_cids;
        active_cids.reserve(static_cast<std::size_t>(nprobe_cap) * static_cast<std::size_t>(qblk));

        std::vector<int> best_ids(static_cast<std::size_t>(nprobe_cap), 0);
        std::vector<float> best_scores(static_cast<std::size_t>(nprobe_cap), 0.0f);

        // ---- Probe mode setup (branch decision made once, outside query loop) ----
        const std::string probe_mode = cfg.eval.linkage_ivf_probe_mode;
        if (probe_mode != "exact" && probe_mode != "hier2" && probe_mode != "hnsw") {
            if (err) {
                *err = "Unsupported eval.linkage.ivf_probe_mode: " + probe_mode + " (expected exact|hier2|hnsw)";
            }
            return false;
        }
        const bool probe_exact = (probe_mode == "exact");
        const bool probe_hier2 = (probe_mode == "hier2");
        const bool probe_hnsw = (probe_mode == "hnsw");

        std::vector<float> root_inv_norm_exact;
        if (probe_exact) {
            std::vector<float> root_norm2(static_cast<std::size_t>(nlist), 1.0f);
            root_inv_norm_exact.assign(static_cast<std::size_t>(nlist), 1.0f);
            for (int cid = 0; cid < nlist; ++cid) {
                const float* c = C_root0.Col(cid);
                double ss = 0.0;
                for (int r = 0; r < d; ++r) {
                    const auto v = static_cast<double>(c[r]);
                    ss += v * v;
                }
                const float norm2 = static_cast<float>(ss);
                root_norm2[static_cast<std::size_t>(cid)] = norm2;
                root_inv_norm_exact[static_cast<std::size_t>(cid)] =
                    1.0f / std::sqrt(std::max(norm2, 1e-20f));
            }
        }

        ivf::Hier2Split hier2;
        int hier2_topL = 1;
        ColMajorMatrix<float> C_coarse;
        std::vector<float> coarse_norm2;
        std::vector<float> root_inv_norm_hier2;
        ColMajorMatrix<float> C_root0_unit;
        if (probe_hier2) {
            const int K1 = ivf::ComputeHier2DefaultK1(nlist);
            const int kmeans_iters = std::max(1, cfg.eval.linkage_ivf_hier2_kmeans_niter);
            const unsigned seed = 12345u;
            const std::filesystem::path cache_dir =
                DefaultHier2SplitCacheDirFromLinkageListDir(std::filesystem::path(linkage_list.dir()));

            // Build normalized centroids (spherical space) and compute a stable hash over them.
            std::uint64_t unit_hash = 0x9e3779b97f4a7c15ULL;
            unit_hash = Mix64(unit_hash, 1ULL); // version
            unit_hash = Mix64(unit_hash, static_cast<std::uint64_t>(d));
            unit_hash = Mix64(unit_hash, static_cast<std::uint64_t>(nlist));

            root_inv_norm_hier2.assign(static_cast<std::size_t>(nlist), 1.0f);
            C_root0_unit.rows = d;
            C_root0_unit.cols = nlist;
            C_root0_unit.data.resize(C_root0.data.size(), 0.0f);
            for (int cid = 0; cid < nlist; ++cid) {
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
                    // Should not happen; treat as cache miss.
                }
                else if (cached.K1 == K1 && cached.K == nlist && !cached.groups.empty()) {
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
                                                   /*kmeans_iters=*/kmeans_iters,
                                                   /*seed=*/seed);
                TryWriteHier2SplitCache(cache_dir, d, nlist, hier2, kmeans_iters, seed, unit_hash);
            }
            else {
                if (!bundle || !bundle->hier2_logged) {
                    LogInfo("[hier2] loaded cached split nlist=" + std::to_string(nlist) +
                        " K1=" + std::to_string(K1) +
                        " iters=" + std::to_string(kmeans_iters) +
                        " dir=" + cache_dir.string());
                }
            }
            hier2_topL = std::max(1, std::min(hier2.K1, cfg.eval.linkage_ivf_hier2_top_coarse));
            // Build coarse centroids in the same normalized space used by exact probe selection.
            std::vector<float> coarse_flat = ivf::BuildCoarseFromHier2(C_root0_unit.data.data(), d, hier2);
            C_coarse.rows = d;
            C_coarse.cols = hier2.K1;
            C_coarse.data = std::move(coarse_flat);
            coarse_norm2.resize(static_cast<std::size_t>(hier2.K1), 0.0f);
            for (int c = 0; c < hier2.K1; ++c) {
                const float* col = C_coarse.Col(c);
                double ss = 0.0;
                for (int r = 0; r < d; ++r) {
                    const auto v = static_cast<double>(col[r]);
                    ss += v * v;
                }
                coarse_norm2[static_cast<std::size_t>(c)] = static_cast<float>(ss);
            }
            // Log group size statistics.
            if (!bundle || !bundle->hier2_logged) {
                int min_sz = nlist, max_sz = 0;
                double sum_sz = 0.0;
                for (int c = 0; c < hier2.K1; ++c) {
                    const int sz = static_cast<int>(hier2.groups[static_cast<std::size_t>(c)].size());
                    min_sz = std::min(min_sz, sz);
                    max_sz = std::max(max_sz, sz);
                    sum_sz += sz;
                }
                LogInfo("[hier2] spherical k-means: nlist=" + std::to_string(nlist) +
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
            }
            else {
                probe_hnsw_index = std::make_shared<ivf::IvfProbeHnswIndex>();
                const std::filesystem::path index_dir =
                    ivf::IvfProbeHnswIndex::DefaultIndexDirFromLinkageListDir(
                        std::filesystem::path(linkage_list.dir()));
                const int M_eff = std::max(2, cfg.eval.linkage_ivf_hnsw_M);
                const int efc_eff = std::max(8, cfg.eval.linkage_ivf_hnsw_ef_construction);
                const int build_threads =
                    (cfg.runtime.omp_threads > 0) ? cfg.runtime.omp_threads : stlq::OmpMaxThreads();
                bool built = false;
                std::string local_err;
                if (!probe_hnsw_index->LoadOrBuildUnitIP(C_root0, index_dir, M_eff, efc_eff, build_threads, &built,
                                                         &local_err)) {
                    if (err) *err = local_err.empty() ? "IVF probe HNSW: load/build failed." : local_err;
                    return false;
                }
                if (bundle) {
                    bundle->ivf_probe_hnsw = probe_hnsw_index;
                }
                if (!bundle || !bundle->hnsw_logged) {
                    LogInfo(std::string("[ivf_hnsw] ") + (built
                                                              ? "building index (first time; will be cached)"
                                                              : "loaded cached index") +
                        " nlist=" + std::to_string(nlist) +
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

        // Query-parallel scan: build per-query probed clusters, preload active clusters once,
        // then scan in `#pragma omp for` over queries.
        std::vector<int> q_cids_flat;
        std::vector<int> q_cids_len;
        std::vector<int> cid_to_active;
        std::vector<stlq::eval::ClusterView> active_views;

        // Pre-allocate coarse GEMM output buffer (K1 × qblk) for hier2 mode; reused across blocks.
        // Sized to qblk (max block size) so inner loops never reallocate.
        // This allocation is preparatory: counted in total wall time, NOT in core/GEMM time.
        ColMajorMatrix<float> xCq_coarse_buf;
        if (probe_hier2) {
            xCq_coarse_buf.rows = hier2.K1;
            xCq_coarse_buf.cols = qblk;
            xCq_coarse_buf.data.resize(static_cast<std::size_t>(hier2.K1) *
                                       static_cast<std::size_t>(qblk), 0.0f);
        }

        for (int q0 = 0; q0 < nquery; q0 += qblk) {
            const int qlen = std::min(qblk, nquery - q0);

            for (int cid : active_cids) {
                active_flag[static_cast<std::size_t>(cid)] = 0;
            }
            active_cids.clear();

            const float* Xq_blk =
                query_dataset_inmem.Xq.data.data() + static_cast<std::size_t>(q0) * static_cast<std::size_t>(d);

            double t_coarse = 0.0;
            double t_root_small = 0.0;
            double t_one = 0.0;
            const double qt_t0 = omp_get_wtime();
            STLQueryTables qt;
            if (probe_exact) {
                // Exact mode: full GEMM (h0 × d × qlen) over all root centroids.
                qt = BuildSTLQueryTables(Xq_blk,
                                         /*ldXq=*/d,
                                         /*d=*/d,
                                         /*qlen=*/qlen,
                                         C_root0, meta_root_small, meta_one,
                                         OmpMaxThreads(),
                                         /*out_coarse_gemm_sec=*/&t_coarse,
                                         /*out_root_small_gemm_sec=*/&t_root_small,
                                         /*out_one_gemm_sec=*/&t_one);
            }
            else if (probe_hier2) {
                // Hier2 mode: table GEMMs (root_small + one) + coarse GEMM (K1 × d × qlen).
                // The table GEMMs are the same as exact mode.
                // The coarse GEMM replaces the expensive h0×d×qlen GEMM with a K1×d×qlen one.
                //
                // Timing discipline (consistent with exact mode):
                //   t_tables  = just root_small + one GemmRaw (inside BuildSTLQueryTablesSkipCoarse)
                //   t_coarse  = just the coarse GemmRaw
                //   qt_t0 block = above GEMMs + ScopedBlasThreads overhead + fine scan
                //   => total_core_qt_wall includes all; total_core_qt_gemm_wall = t_coarse + t_tables only.
                qt = BuildSTLQueryTablesSkipCoarse(Xq_blk,
                                                   /*ldXq=*/d,
                                                   /*d=*/d,
                                                   /*qlen=*/qlen,
                                                   /*h0=*/nlist,
                                                   meta_root_small, meta_one,
                                                   OmpMaxThreads(),
                                                   /*out_root_small_gemm_sec=*/&t_root_small,
                                                   /*out_one_gemm_sec=*/&t_one);
                // Adjust cols in the pre-allocated buffer to match this block (last block may be shorter).
                xCq_coarse_buf.cols = qlen;
                {
                    // ScopedBlasThreads setup is outside t0_c; only GemmRaw is timed,
                    // matching exact mode where BuildSTLQueryTables times only the GemmRaw calls.
                    ScopedBlasThreads blas_scope(OmpMaxThreads());
                    const double t0_c = omp_get_wtime();
                    GemmRaw(/*trans_a=*/true, /*trans_b=*/false,
                                        /*m=*/hier2.K1, /*n=*/qlen, /*k=*/d,
                                        /*alpha=*/1.0f,
                                        /*A=*/C_coarse.data.data(), /*lda=*/C_coarse.rows,
                                        /*B=*/Xq_blk, /*ldb=*/d,
                                        /*beta=*/0.0f,
                                        /*C=*/xCq_coarse_buf.data.data(), /*ldc=*/xCq_coarse_buf.rows);
                    t_coarse = omp_get_wtime() - t0_c;
                }
            }
            else {
                // HNSW mode: table GEMMs only; root0 dot products are filled sparsely after HNSW selection.
                qt = BuildSTLQueryTablesSkipCoarse(Xq_blk,
                                                   /*ldXq=*/d,
                                                   /*d=*/d,
                                                   /*qlen=*/qlen,
                                                   /*h0=*/nlist,
                                                   meta_root_small, meta_one,
                                                   OmpMaxThreads(),
                                                   /*out_root_small_gemm_sec=*/&t_root_small,
                                                   /*out_one_gemm_sec=*/&t_one);
            }
            total_core_qt_wall += omp_get_wtime() - qt_t0;
            const double t_tables = t_root_small + t_one;
            total_core_qt_gemm_wall += (t_coarse + t_tables);
            total_core_qt_coarse_gemm_wall += t_coarse;
            total_core_qt_root_small_gemm_wall += t_root_small;
            total_core_qt_one_gemm_wall += t_one;
            if (profile_timing) {
                total_coarse_gemm += t_coarse;
                total_table_gemm += t_tables;
            }

            std::vector<TopKHeap> topk;
            topk.reserve(static_cast<std::size_t>(qlen));
            for (int qi = 0; qi < qlen; ++qi) topk.emplace_back(k);

            q_cids_flat.assign(static_cast<std::size_t>(qlen) * static_cast<std::size_t>(nprobe_cap), 0);
            q_cids_len.assign(static_cast<std::size_t>(qlen), 0);

            // Reusable buffer for coarse group ids (size = hier2_topL); hoisted outside qi loop.
            std::vector<int> top_coarse_ids(static_cast<std::size_t>(hier2_topL), 0);

            // Reusable buffer for fine candidates in hier2 mode (hoisted outside qi loop to
            // avoid repeated heap allocations per query – previously a major perf bottleneck).
            struct FineCandidate
            {
                float score;
                float raw_dot;
                int id;
            };
            std::vector<FineCandidate> fine_cands;

            // Probe selection timing: covers SelectTopClusters (exact) or coarse-select + fine
            // dot-product scan + nth_element+sort (hier2). Both modes timed with the same accumulator
            // so core QPS = qt_gemm + probe_sel + scan_topk is consistent across modes.
            const double probe_sel_t0 = omp_get_wtime();
            // Per-query selection is independent in both exact/hier2 modes; parallelize over queries.
#pragma omp parallel
            {
                std::vector<int> best_ids_local(static_cast<std::size_t>(nprobe_cap), 0);
                std::vector<float> best_scores_local(static_cast<std::size_t>(nprobe_cap), 0.0f);
                std::vector<float> probe_scores_local;
                std::vector<int> top_coarse_ids_local;
                std::vector<FineCandidate> fine_cands_local;
                if (probe_hier2) {
                    top_coarse_ids_local.assign(static_cast<std::size_t>(hier2_topL), 0);
                }
                else if (probe_exact) {
                    probe_scores_local.assign(static_cast<std::size_t>(nlist),
                                              -std::numeric_limits<float>::infinity());
                }

#pragma omp for schedule(static)
                for (int qi = 0; qi < qlen; ++qi) {
                    int nprobe = 0;
                    if (probe_exact) {
                        const float* scores = qt.xCq_root0.Col(qi);
                        for (int cid = 0; cid < nlist; ++cid) {
                            probe_scores_local[static_cast<std::size_t>(cid)] =
                                scores[cid] * root_inv_norm_exact[static_cast<std::size_t>(cid)];
                        }
                        nprobe = ivf::SelectTopClustersByScore(probe_scores_local.data(), nlist, nprobe_cap,
                                                               best_ids_local.data(), best_scores_local.data());
                    }
                    else if (probe_hier2) {
                        const float* dot_coarse = xCq_coarse_buf.Col(qi);
                        std::fill(top_coarse_ids_local.begin(), top_coarse_ids_local.end(), 0);
                        const int got_coarse = ivf::SelectTopCoarseByScore(
                            dot_coarse, nullptr, hier2.K1, hier2_topL, top_coarse_ids_local.data());

                        const float* q = Xq_blk + static_cast<std::size_t>(qi) * static_cast<std::size_t>(d);
                        fine_cands_local.clear();
                        for (int t = 0; t < got_coarse; ++t) {
                            const int cg = top_coarse_ids_local[static_cast<std::size_t>(t)];
                            const auto& members = hier2.groups[static_cast<std::size_t>(cg)];
                            for (int fid : members) {
                                const float* c = C_root0.Col(fid);
                                float dot = 0.0f;
#pragma omp simd reduction(+:dot)
                                for (int r = 0; r < d; ++r) dot += q[r] * c[r];
                                fine_cands_local.push_back(FineCandidate{
                                    dot * root_inv_norm_hier2[static_cast<std::size_t>(fid)],
                                    dot,
                                    fid
                                });
                            }
                        }

                        const int want = std::min(nprobe_cap, static_cast<int>(fine_cands_local.size()));
                        if (want > 0) {
                            const auto better = [](const FineCandidate& a, const FineCandidate& b)
                            {
                                return a.score > b.score;
                            };
                            if (want < static_cast<int>(fine_cands_local.size())) {
                                std::nth_element(fine_cands_local.begin(), fine_cands_local.begin() + want,
                                                 fine_cands_local.end(), better);
                            }
                            std::sort(fine_cands_local.begin(), fine_cands_local.begin() + want, better);
                        }
                        nprobe = want;
                        for (int j = 0; j < nprobe; ++j) {
                            best_ids_local[static_cast<std::size_t>(j)] = fine_cands_local[static_cast<std::size_t>(j)].
                                id;
                            best_scores_local[static_cast<std::size_t>(j)] = fine_cands_local[static_cast<std::size_t>(
                                j)].raw_dot;
                        }

                        // Fill sparse xCq_root0 entries with raw root dots; scan kernels expect exact dot semantics here.
                        float* root0_col = qt.xCq_root0.Col(qi);
                        for (int j = 0; j < nprobe; ++j) {
                            root0_col[best_ids_local[static_cast<std::size_t>(j)]] =
                                best_scores_local[static_cast<std::size_t>(j)];
                        }
                    }
                    else {
                        // HNSW probe: select centroids by IP over unit-normalized centroids.
                        const float* q = Xq_blk + static_cast<std::size_t>(qi) * static_cast<std::size_t>(d);
                        const int want = std::min(nprobe_cap, nlist);
                        (void)probe_hnsw_index->SearchTopK(q, want,
                                                           best_ids_local.data(),
                                                           /*out_dist=*/nullptr,
                                                           /*err=*/nullptr);
                        float* root0_col = qt.xCq_root0.Col(qi);
                        nprobe = 0;
                        for (int j = 0; j < want; ++j) {
                            const int cid = best_ids_local[static_cast<std::size_t>(j)];
                            if (cid < 0 || cid >= nlist) continue;
                            const float* c = C_root0.Col(cid);
                            float dot = 0.0f;
#pragma omp simd reduction(+:dot)
                            for (int r = 0; r < d; ++r) dot += q[r] * c[r];
                            best_ids_local[static_cast<std::size_t>(nprobe)] = cid;
                            best_scores_local[static_cast<std::size_t>(nprobe)] = dot;
                            root0_col[cid] = dot;
                            ++nprobe;
                        }
                    }

                    q_cids_len[static_cast<std::size_t>(qi)] = nprobe;
                    int* dst = q_cids_flat.data() +
                        static_cast<std::size_t>(qi) * static_cast<std::size_t>(nprobe_cap);
                    for (int t = 0; t < nprobe; ++t) {
                        dst[t] = best_ids_local[static_cast<std::size_t>(t)];
                    }
                }
            }

            // Deduplicate active cluster ids (serial) for preload.
            for (int qi = 0; qi < qlen; ++qi) {
                const int nprobe = q_cids_len[static_cast<std::size_t>(qi)];
                const int* src = q_cids_flat.data() +
                    static_cast<std::size_t>(qi) * static_cast<std::size_t>(nprobe_cap);
                for (int t = 0; t < nprobe; ++t) {
                    const int cid = src[t];
                    if (active_flag[static_cast<std::size_t>(cid)] == 0) {
                        active_flag[static_cast<std::size_t>(cid)] = 1;
                        active_cids.push_back(cid);
                    }
                }
            }
            total_core_probe_sel_wall += omp_get_wtime() - probe_sel_t0;

            cid_to_active.assign(static_cast<std::size_t>(nlist), -1);
            active_views.resize(static_cast<std::size_t>(active_cids.size()));
            for (std::size_t i = 0; i < active_cids.size(); ++i) {
                cid_to_active[static_cast<std::size_t>(active_cids[i])] = static_cast<int>(i);
            }
            for (std::size_t i = 0; i < active_cids.size(); ++i) {
                std::string local_err;
                const int cid = active_cids[i];
                stlq::eval::ClusterProvider::PrepStats prep{};
                if (!provider.GetCluster(cid, &active_views[i], &prep, &local_err)) {
                    if (err)
                        *err = local_err.empty()
                                   ? "EvaluateRecallLinkageIvfFromDisk(lookup): cluster load failed."
                                   : local_err;
                    return false;
                }
                // Skip empty clusters (valid when nlist is large); they contribute no DB points.
                if (active_views[i].n_real <= 0 || active_views[i].nc <= 0) {
                    cid_to_active[static_cast<std::size_t>(cid)] = -1;
                    continue;
                }
                // NOTE: louds/coeff decode times are NOT accumulated here for the preload path.
                // In preload mode: all clusters are warm-cache hits (prep times ≈ 0);
                //   actual cost is measured by BenchmarkDecodeWall below.
                // In lazy mode: the block below accumulates times from cache-miss loads.
                if (is_lazy_mode) {
                    if (prep.parent_louds_decode_sec > 0.0) {
                        // parent_louds_decode_sec = CPU-only (deser+decode+slice); read_sec = I/O-only.
                        total_parent_louds_decode += prep.parent_louds_decode_sec; // CPU-only
                        total_parent_louds_io_wall += prep.parent_louds_decode_sec // total wall
                            + prep.parent_louds_read_sec; //  = CPU + I/O
                        ++lazy_louds_decoded_count;
                    }
                    if (prep.coeff_decode_sec > 0.0 || prep.coeff_io_sec > 0.0) {
                        total_coeff_decode += prep.coeff_decode_sec; // CPU-only Huffman decode
                        total_coeff_io_wall += prep.coeff_io_sec; // disk read I/O wall (separate)
                        ++lazy_coeff_decoded_count;
                    }
                    if (bundle && (prep.parent_louds_decode_sec > 0.0 ||
                        prep.coeff_decode_sec > 0.0 ||
                        prep.coeff_io_sec > 0.0)) {
                        // Cache-miss prep stats are non-zero only on the first load of a cid in this
                        // provider session, so this append naturally stays unique across repeat calls.
                        bundle->lazy_loaded_cids.push_back(cid);
                    }
                }
                if (profile_timing) {
                    total_parent_louds_read += prep.parent_louds_read_sec;
                    total_parent_louds_deser += prep.parent_louds_deser_sec;
                    total_parent_louds_decode_parent += prep.parent_louds_decode_parent_sec;
                    total_parent_louds_slice += prep.parent_louds_slice_sec;
                    total_norm += prep.norm_prep_sec;
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                    if (norm_provider_cuda && prep.norm_prep_sec > 0.0) {
                        const auto& ns = norm_provider_cuda->last_stats();
                        if (ns.used_gpu) {
                            ++total_norm_gpu_used;
                        }
                        else {
                            ++total_norm_gpu_fallback;
                            if (ns.fallback_reason >= 0 && ns.fallback_reason <= 4) {
                                ++total_norm_gpu_fallback_reason[ns.fallback_reason];
                            }
                            if (ns.fallback_reason == 2 && norm_gpu_first_too_large_cid < 0) {
                                norm_gpu_first_too_large_cid = active_views[i].cid;
                                norm_gpu_first_too_large_nreal = active_views[i].n_real;
                                norm_gpu_first_too_large_nc = active_views[i].nc;
                            }
                        }
                        total_norm_cpu_c0dot += ns.cpu_c0dot_sec;
                        total_norm_gpu_h2d += ns.h2d_sec;
                        total_norm_gpu_kernel += ns.kernel_sec;
                        total_norm_gpu_d2h += ns.d2h_sec;
                    }
#endif
                }
            }

            // Release raw Huffman payloads (lens/payload) from all cached CoeffEntry objects.
            // They are only needed for BenchmarkDecodeWall; scan uses decoded q_layer_major.
            // In lazy mode, delay release until after the end-of-eval CPU-only benchmark so the
            // raw payloads of freshly loaded clusters remain available.
            if (!is_lazy_mode) {
                provider.ReleaseCoeffRawPayloads();
            }

            const double core_scan_topk_t0 = omp_get_wtime();

            // ---- Scan + TopK (dispatch) ----
            // NOTE: `profile_timing` comes from config (cfg.large.profile_timing).
            // It controls extra profile breakdown bookkeeping and is independent from the `timing` output pointer.
            ScanTopKProfileSums scan_sums;
            if (profile_timing) {
                scan_sums.enabled = true;
                scan_sums.total_scan_kernel = &total_scan_kernel;
                scan_sums.total_topk_finalize = &total_topk_finalize;
                scan_sums.total_topk_finalize_worker = &total_topk_finalize_worker;
                scan_sums.total_scan_scale_tables = &total_scan_scale_tables;
                scan_sums.total_scan_roots_init = &total_scan_roots_init;
                scan_sums.total_scan_roots_layers = &total_scan_roots_layers;
                scan_sums.total_scan_linkage = &total_scan_linkage;
                scan_sums.total_scan_push = &total_scan_push;
                scan_sums.total_scan_push_dist = &total_scan_push_dist;
                scan_sums.total_scan_push_heap = &total_scan_push_heap;
                scan_sums.total_gpu_scan_pack_h2d = &total_gpu_scan_pack_h2d;
                scan_sums.total_gpu_scan_host_pack = &total_gpu_scan_host_pack;
                scan_sums.total_gpu_scan_alloc = &total_gpu_scan_alloc;
                scan_sums.total_gpu_scan_tables_h2d = &total_gpu_scan_tables_h2d;
                scan_sums.total_gpu_scan_kernel = &total_gpu_scan_kernel;
                scan_sums.total_gpu_scan_k_roots = &total_gpu_scan_k_roots;
                scan_sums.total_gpu_scan_k_depth = &total_gpu_scan_k_depth;
                scan_sums.total_gpu_scan_k_topk = &total_gpu_scan_k_topk;
                scan_sums.total_gpu_scan_out_d2h = &total_gpu_scan_out_d2h;
                scan_sums.total_gpu_scan_stream_sync = &total_gpu_scan_stream_sync;
                scan_sums.total_gpu_scan_cpu_fallback = &total_gpu_scan_cpu_fallback;
                scan_sums.total_gpu_scan_tasks_total = &total_gpu_scan_tasks_total;
                scan_sums.total_gpu_scan_tasks_gpu = &total_gpu_scan_tasks_gpu;
                scan_sums.total_gpu_scan_clusters_total = &total_gpu_scan_clusters_total;
                scan_sums.total_gpu_scan_clusters_gpu = &total_gpu_scan_clusters_gpu;
                scan_sums.total_gpu_scan_kernel_max_nc = &total_gpu_scan_kernel_max_nc;
                scan_sums.total_gpu_scan_cache_slots = &total_gpu_scan_cache_slots;
                scan_sums.total_gpu_scan_cache_calls = &total_gpu_scan_cache_calls;
                scan_sums.total_gpu_scan_cache_enabled_calls = &total_gpu_scan_cache_enabled_calls;
                scan_sums.total_gpu_scan_cache_hits = &total_gpu_scan_cache_hits;
                scan_sums.total_gpu_scan_cache_misses = &total_gpu_scan_cache_misses;
                scan_sums.total_gpu_scan_cache_upload_clusters = &total_gpu_scan_cache_upload_clusters;
                scan_sums.total_gpu_scan_cache_upload_bytes = &total_gpu_scan_cache_upload_bytes;
                scan_sums.total_gpu_scan_task_cluster_idx_bytes = &total_gpu_scan_task_cluster_idx_bytes;
                scan_sums.total_gpu_scan_out_d2h_bytes = &total_gpu_scan_out_d2h_bytes;
                scan_sums.gpu_scan_failed_any = &gpu_scan_failed_any;
                scan_sums.gpu_scan_first_err = &gpu_scan_first_err;
            }

            if (use_coeff_codec) {
                RunScanTopK_Int8Coeff(
                    cfg, linkage_list, qt,
                    m, m_codes,
                    offsets_root_small, meta_one,
                    root_small_total_cols, one_total_cols,
                    scan_coeff_fn,
                    q_cids_len, q_cids_flat, cid_to_active, active_views,
                    q0, qlen, nprobe_cap, k,
                    &topk, out,
                    &scan_sums);
            }
            else {
                RunScanTopK_FloatCoeffCpu(
                    qt,
                    offsets_root_small, meta_one,
                    q_cids_len, q_cids_flat, cid_to_active, active_views,
                    q0, qlen, nprobe_cap,
                    &topk, out,
                    profile_timing ? &scan_sums : nullptr);
            }

            total_core_scan_topk_wall += omp_get_wtime() - core_scan_topk_t0;
        }

        // ---- Inject benchmark decode times into summary accumulators ----
        // Preload mode: report one full all-cluster decode pass directly (no per-query scaling).
        if (!is_lazy_mode && bundle && bundle->bench_done) {
            total_parent_louds_decode = bundle->bench_louds_wall_avg_sec;
            total_coeff_decode = bundle->bench_coeff_wall_avg_sec;
            if (profile_timing) {
                total_parent_louds_deser = bundle->bench_profile_louds_deser_sec;
                total_parent_louds_decode_parent = bundle->bench_profile_louds_decode_parent_sec;
                total_parent_louds_slice = bundle->bench_profile_louds_slice_sec;
            }
        }

        // Lazy mode: keep the actual cold-load wall time from PrepStats, but benchmark the already
        // loaded clusters once to obtain a CPU-only decode wall time comparable to preload mode.
        if (is_lazy_mode && bundle) {
            if (total_parent_louds_io_wall > 0.0) {
                bundle->lazy_louds_total_wall_sec = total_parent_louds_io_wall;
            }
            if (use_coeff_codec) {
                const double coeff_total_wall_sec = total_coeff_io_wall + total_coeff_decode;
                if (coeff_total_wall_sec > 0.0) {
                    bundle->lazy_coeff_total_wall_sec = coeff_total_wall_sec;
                }
            }
            if (!bundle->lazy_bench_done && !bundle->lazy_loaded_cids.empty()) {
                eval::ClusterProvider::BenchmarkResult lazy_bench_res{};
                std::string lazy_bench_err;
                if (!provider.BenchmarkDecodeWall(bundle->lazy_loaded_cids, bench_times,
                                                  /*request_profile_breakdown=*/false,
                                                  &lazy_bench_res, &lazy_bench_err)) {
                    if (err) *err = "Linkage recall: lazy BenchmarkDecodeWall failed: " + lazy_bench_err;
                    return false;
                }
                bundle->lazy_louds_cpu_wall_sec = lazy_bench_res.louds_wall_avg_sec;
                bundle->lazy_coeff_cpu_wall_sec = lazy_bench_res.coeff_wall_avg_sec;
                bundle->lazy_bench_done = true;
            }
            if (use_coeff_codec) {
                provider.ReleaseCoeffRawPayloads();
            }
        }

        if (bundle) {
            bundle->parent_storage_stats = provider.GetParentStorageStats();
        }

        double report_parent_louds_cpu_sec = total_parent_louds_decode;
        [[maybe_unused]] double report_coeff_cpu_sec = total_coeff_decode;
        double report_parent_louds_total_wall_sec = is_lazy_mode
                                                        ? total_parent_louds_io_wall
                                                        : total_parent_louds_decode;
        double report_coeff_total_wall_sec = is_lazy_mode
                                                 ? (total_coeff_io_wall + total_coeff_decode)
                                                 : total_coeff_decode;
        double report_preload_total_wall_sec = 0.0;
        if (is_lazy_mode && bundle) {
            if (bundle->lazy_bench_done) {
                report_parent_louds_cpu_sec = bundle->lazy_louds_cpu_wall_sec;
                if (use_coeff_codec) {
                    report_coeff_cpu_sec = bundle->lazy_coeff_cpu_wall_sec;
                }
            }
            if (bundle->lazy_louds_total_wall_sec > 0.0) {
                report_parent_louds_total_wall_sec = bundle->lazy_louds_total_wall_sec;
            }
            if (use_coeff_codec) {
                if (bundle->lazy_coeff_total_wall_sec > 0.0) {
                    report_coeff_total_wall_sec = bundle->lazy_coeff_total_wall_sec;
                }
            }
        }
        else if (bundle && bundle->preload_total_wall_sec > 0.0) {
            report_preload_total_wall_sec = bundle->preload_total_wall_sec;
            report_parent_louds_total_wall_sec = bundle->preload_total_wall_sec;
            if (use_coeff_codec) {
                report_coeff_total_wall_sec = bundle->preload_total_wall_sec;
            }
        }

        if (profile_timing) {
            if (!cfg.eval.bench_quiet) {
                LogInfo("---- Disk IVF linkage timing breakdown (profile) ----");
                if (qt_rotate_wall_sec > 0.0) {
                    LogInfo("Disk IVF linkage qt_rotate time (wall): " + std::to_string(qt_rotate_wall_sec) + "s");
                }
                LogInfo(
                    "Disk IVF linkage qt_build coarse GEMM time (wall): " + std::to_string(total_coarse_gemm) + "s");
                LogInfo("Disk IVF linkage qt_build table GEMM time (wall): " + std::to_string(total_table_gemm) + "s");
            }
            if (cfg.eval.linkage_parent_louds_enable) {
                if (!store_has_parent_louds) {
                    if (!cfg.eval.bench_quiet) {
                        LogInfo(
                            "Disk IVF linkage parent LOUDS: requested but not present in store (meta.store_parent_louds=0)");
                    }
                }
                else {
                    if (!cfg.eval.bench_quiet && (!bundle || !bundle->louds_timing_reported)) {
                        const std::string louds_mode_note = is_lazy_mode
                                                                ? ("(cpu-only bench wall, clusters=" + std::to_string(
                                                                    static_cast<int>(bundle
                                                                        ? bundle->lazy_loaded_cids.size()
                                                                        : lazy_louds_decoded_count)) + ")")
                                                                : ("(preload all-cluster bench, clusters=" +
                                                                    std::to_string(
                                                                        bundle ? bundle->bench_n_clusters : 0) +
                                                                    ", bench_times=" + std::to_string(bench_times) +
                                                                    ")");
                        const std::string louds_io_suffix = is_lazy_mode
                                                                ? ("  (random-io+decode wall: " + std::to_string(
                                                                    report_parent_louds_total_wall_sec) + "s)")
                                                                : ((report_preload_total_wall_sec > 0.0)
                                                                       ? ("  (shared preload io+decode wall: " +
                                                                           std::to_string(report_preload_total_wall_sec)
                                                                           + "s)")
                                                                       : std::string());
                        LogInfo("Disk IVF linkage parent LOUDS decode CPU time " + louds_mode_note + ": " +
                            std::to_string(report_parent_louds_cpu_sec) + "s" + louds_io_suffix);
                        LogInfo("Disk IVF linkage parent LOUDS breakdown(s, CPU thread-sum, one profile pass): read=" +
                            std::to_string(total_parent_louds_read) +
                            " deser=" + std::to_string(total_parent_louds_deser) +
                            " decode=" + std::to_string(total_parent_louds_decode_parent) +
                            " slice=" + std::to_string(total_parent_louds_slice));
                    }
                }
            }
            else if (store_has_parent_louds) {
                if (!cfg.eval.bench_quiet) {
                    LogInfo(
                        "Disk IVF linkage parent LOUDS: disabled by config (eval.linkage.parent_louds_enable=false)");
                }
            }
            if (use_coeff_codec) {
                if (!cfg.eval.bench_quiet && (!bundle || !bundle->louds_timing_reported)) {
                    const std::string huffman_mode_note = is_lazy_mode
                                                              ? ("(cpu-only bench wall, clusters=" + std::to_string(
                                                                  static_cast<int>(bundle
                                                                      ? bundle->lazy_loaded_cids.size()
                                                                      : lazy_coeff_decoded_count)) + ")")
                                                              : ("(preload all-cluster bench, clusters=" +
                                                                  std::to_string(
                                                                      bundle ? bundle->bench_n_clusters : 0) +
                                                                  ", bench_times=" + std::to_string(bench_times) + ")");
                    const std::string huffman_io_suffix = is_lazy_mode
                                                              ? ("  (random-io+decode wall: " + std::to_string(
                                                                  report_coeff_total_wall_sec) + "s)")
                                                              : ((report_preload_total_wall_sec > 0.0)
                                                                     ? ("  (shared preload io+decode wall: " +
                                                                         std::to_string(report_preload_total_wall_sec) +
                                                                         "s)")
                                                                     : std::string());
                    LogInfo("Disk IVF linkage Huffman decode CPU time " + huffman_mode_note + ": " +
                        std::to_string(report_coeff_cpu_sec) + "s" + huffman_io_suffix);
                }
            }
            if (!cfg.eval.bench_quiet) {
                LogInfo(
                    "Disk IVF linkage norm+recon time (wall, sum over clusters): " + std::to_string(total_norm) + "s");
                LogInfo("Disk IVF linkage scan kernel time (wall): " + std::to_string(total_scan_kernel) + "s");
            }
            if (use_coeff_codec) {
                const bool want_gpu_scan = cfg.eval.linkage_gpu_scan_enable && cfg.runtime.use_cuda &&
                    !eval::IsDiskNorm2ModeLut(cfg.eval.disk_norm2_mode);
                if (!want_gpu_scan) {
                    if (!cfg.eval.bench_quiet) {
                        LogInfo("Disk IVF linkage scan breakdown (sum over threads): "
                            "scale_tables=" + std::to_string(total_scan_scale_tables) +
                            " roots_init=" + std::to_string(total_scan_roots_init) +
                            " roots_layers=" + std::to_string(total_scan_roots_layers) +
                            " linkage=" + std::to_string(total_scan_linkage) +
                            " push=" + std::to_string(total_scan_push) +
                            " push_dist=" + std::to_string(total_scan_push_dist) +
                            " push_heap=" + std::to_string(total_scan_push_heap));

                        // The production CPU path fuses distance generation with online heap
                        // maintenance.  Profiling separates finalization into its own parallel
                        // phase, then apportions the fused scan-region wall time using active
                        // worker time.  This produces an additive logical Scan/TopK breakdown
                        // without materializing every candidate distance.  It is intentionally
                        // labelled normalized rather than directly measured wall time.
                        const double scan_worker =
                            total_scan_scale_tables + total_scan_roots_init + total_scan_roots_layers +
                            total_scan_linkage + std::max(0.0, total_scan_push - total_scan_push_heap);
                        const double online_topk_worker = total_scan_push_heap;
                        const double fused_worker = scan_worker + online_topk_worker;
                        const double online_topk_wall =
                            (fused_worker > 0.0)
                                ? total_scan_kernel * online_topk_worker / fused_worker
                                : 0.0;
                        const double logical_scan_wall = std::max(0.0, total_scan_kernel - online_topk_wall);
                        const double logical_topk_wall = online_topk_wall + total_topk_finalize;
                        LogInfo("Disk IVF linkage topk worker breakdown (sum over threads): online_heap=" +
                            std::to_string(online_topk_worker) +
                            " finalize=" + std::to_string(total_topk_finalize_worker));
                        LogInfo("Disk IVF linkage logical scan/topk breakdown (normalized wall; "
                            "topk=online_heap+finalize): scan=" + std::to_string(logical_scan_wall) +
                            " topk=" + std::to_string(logical_topk_wall) +
                            " total=" + std::to_string(logical_scan_wall + logical_topk_wall));
                    }
                }
                else {
                    if (!cfg.eval.bench_quiet) {
                        LogInfo(
                            "Disk IVF linkage scan note: eval.linkage.gpu_scan_enable=true (CPU breakdown is not collected).");
                    }
                }
            }
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
            if (cfg.eval.linkage_gpu_norm_enable && cfg.runtime.use_cuda && !cfg.eval.bench_quiet) {
                LogInfo("Disk IVF linkage norm(gpu) breakdown(s): cpu_c0dot=" + std::to_string(total_norm_cpu_c0dot) +
                    " h2d=" + std::to_string(total_norm_gpu_h2d) +
                    " kernel=" + std::to_string(total_norm_gpu_kernel) +
                    " d2h=" + std::to_string(total_norm_gpu_d2h) +
                    " used=" + std::to_string(total_norm_gpu_used) +
                    " fallback=" + std::to_string(total_norm_gpu_fallback));
                LogInfo("Disk IVF linkage norm(gpu) fallback reasons: too_large=" +
                    std::to_string(total_norm_gpu_fallback_reason[2]) +
                    " unsupported=" + std::to_string(total_norm_gpu_fallback_reason[3]) +
                    " gpu_error=" + std::to_string(total_norm_gpu_fallback_reason[4]));
                if (total_norm_gpu_fallback_reason[2] > 0 && norm_gpu_first_too_large_cid >= 0) {
                    LogInfo("Disk IVF linkage norm(gpu) first too_large cluster: cid=" +
                        std::to_string(norm_gpu_first_too_large_cid) +
                        " n_real=" + std::to_string(norm_gpu_first_too_large_nreal) +
                        " nc=" + std::to_string(norm_gpu_first_too_large_nc) +
                        " limit_nc=" + std::to_string(cfg.eval.linkage_gpu_scan_max_nc));
                }
            }
#endif
            if (use_coeff_codec) {
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                if (cfg.eval.linkage_gpu_scan_enable && cfg.runtime.use_cuda &&
                    !eval::IsDiskNorm2ModeLut(cfg.eval.disk_norm2_mode) &&
                    total_gpu_scan_tasks_total > 0) {
                    LogInfo("Disk IVF linkage scan(gpu) breakdown(s): host_pack=" + std::to_string(
                            total_gpu_scan_host_pack) +
                        " alloc=" + std::to_string(total_gpu_scan_alloc) +
                        " pack_h2d=" + std::to_string(total_gpu_scan_pack_h2d) +
                        " tables_h2d=" + std::to_string(total_gpu_scan_tables_h2d) +
                        " kernel=" + std::to_string(total_gpu_scan_kernel) +
                        " k_roots=" + std::to_string(total_gpu_scan_k_roots) +
                        " k_depth=" + std::to_string(total_gpu_scan_k_depth) +
                        " k_topk=" + std::to_string(total_gpu_scan_k_topk) +
                        " out_d2h=" + std::to_string(total_gpu_scan_out_d2h) +
                        " out_d2h_mb=" + std::to_string(
                            static_cast<double>(total_gpu_scan_out_d2h_bytes) / (1024.0 * 1024.0)) +
                        " wall_sync=" + std::to_string(total_gpu_scan_stream_sync) +
                        " cpu_fallback=" + std::to_string(total_gpu_scan_cpu_fallback) +
                        " clusters_gpu=" + std::to_string(total_gpu_scan_clusters_gpu) +
                        "/" + std::to_string(total_gpu_scan_clusters_total) +
                        " kernel_max_nc=" + std::to_string(total_gpu_scan_kernel_max_nc) +
                        " cache_slots=" + std::to_string(total_gpu_scan_cache_slots) +
                        " cache_upload=" + std::to_string(total_gpu_scan_cache_upload_clusters) +
                        " cache_upload_mb=" + std::to_string(
                            static_cast<double>(total_gpu_scan_cache_upload_bytes) / (1024.0 * 1024.0)) +
                        " cache_hits=" + std::to_string(total_gpu_scan_cache_hits) +
                        " cache_misses=" + std::to_string(total_gpu_scan_cache_misses) +
                        " cache_calls=" + std::to_string(total_gpu_scan_cache_calls) +
                        " cache_enabled_calls=" + std::to_string(total_gpu_scan_cache_enabled_calls) +
                        " task_idx_mb=" + std::to_string(
                            static_cast<double>(total_gpu_scan_task_cluster_idx_bytes) / (1024.0 * 1024.0)) +
                        " tasks_gpu=" + std::to_string(total_gpu_scan_tasks_gpu) +
                        "/" + std::to_string(total_gpu_scan_tasks_total));
                }
                if (cfg.eval.linkage_gpu_scan_enable && cfg.runtime.use_cuda &&
                    !eval::IsDiskNorm2ModeLut(cfg.eval.disk_norm2_mode) &&
                    total_gpu_scan_tasks_total == 0) {
                    const int device_max_nc = stlq::eval::cuda::DiskLinkageGpuScanMaxNcSupported();
                    if (gpu_scan_failed_any) {
                        LogInfo("Disk IVF linkage scan(gpu): failed; falling back to CPU (gpu_scan_max_nc=" +
                            std::to_string(cfg.eval.linkage_gpu_scan_max_nc) +
                            " device_max_nc=" + std::to_string(device_max_nc) +
                            " err=\"" + gpu_scan_first_err + "\")");
                    }
                    else {
                        LogInfo("Disk IVF linkage scan(gpu): no eligible tasks (gpu_scan_max_nc=" +
                            std::to_string(cfg.eval.linkage_gpu_scan_max_nc) +
                            " device_max_nc=" + std::to_string(device_max_nc) + ")");
                    }
                }
#endif
            }
            if (!cfg.eval.bench_quiet) {
                LogInfo("Disk IVF linkage topk finalize time (wall): " + std::to_string(total_topk_finalize) + "s");
                LogInfo("Disk IVF linkage scan+topk time (wall): " +
                    std::to_string(total_scan_kernel + total_topk_finalize) + "s");
                LogInfo("Disk IVF linkage core time (accounted, qt_rotate+qt_gemm+probe_sel+scan+topk): " +
                    std::to_string(qt_rotate_wall_sec + total_coarse_gemm + total_table_gemm +
                        total_core_probe_sel_wall + total_scan_kernel + total_topk_finalize) + "s");
            }
            const double qt_other = total_core_qt_wall - total_core_qt_gemm_wall;
            if (!cfg.eval.bench_quiet && qt_other > 1e-9) {
                LogInfo("Disk IVF linkage qt_build other time (wall): " + std::to_string(qt_other) + "s");
            }
        }
        else {
            // (wall-time core metrics are printed below for both profile/non-profile modes)
        }

        if (profile_timing) {
            if (!cfg.eval.bench_quiet) {
                LogInfo("---- Disk IVF linkage timing summary (wall/core) ----");
            }
        }
        // core = qt_rotate + qt_gemm + probe_sel + scan_topk
        // probe_sel: exact => SelectTopClusters; hier2 => SelectTopCoarse + fine dot scan + partial_sort.
        // Both modes use the same accumulator so QPS is comparable across modes.
        const double core_wall_sec = qt_rotate_wall_sec + total_core_qt_gemm_wall + total_core_probe_sel_wall +
            total_core_scan_topk_wall;
        if (timing) {
            timing->core_wall_sec = core_wall_sec;
            timing->qt_gemm_wall_sec = total_core_qt_gemm_wall;
            // qt_rotate_wall_sec is an INPUT (set by caller); do NOT overwrite it.
            timing->qt_coarse_gemm_wall_sec = total_core_qt_coarse_gemm_wall;
            timing->qt_root_small_gemm_wall_sec = total_core_qt_root_small_gemm_wall;
            timing->qt_one_gemm_wall_sec = total_core_qt_one_gemm_wall;
            timing->qt_other_wall_sec = (total_core_qt_wall - total_core_qt_gemm_wall);
            timing->probe_sel_wall_sec = total_core_probe_sel_wall;
            timing->scan_topk_wall_sec = total_core_scan_topk_wall;
            timing->louds_wall_sec = report_parent_louds_total_wall_sec;
            timing->louds_cpu_sec = report_parent_louds_cpu_sec;
            timing->huffman_wall_sec = report_coeff_total_wall_sec;
            timing->huffman_cpu_sec = report_coeff_cpu_sec;
            if (session) session->louds_load_mode = is_lazy_mode ? "lazy" : "preload";
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
            if (cfg.eval.linkage_parent_louds_enable && store_has_parent_louds && (!bundle || !bundle->
                louds_timing_reported)) {
                const std::string louds_mode_note = is_lazy_mode
                                                        ? ("(cpu-only bench wall, clusters=" + std::to_string(
                                                            static_cast<int>(bundle
                                                                                 ? bundle->lazy_loaded_cids.size()
                                                                                 : lazy_louds_decoded_count)) + ")")
                                                        : ("(preload all-cluster bench, clusters=" + std::to_string(
                                                                bundle ? bundle->bench_n_clusters : 0) +
                                                            ", bench_times=" + std::to_string(bench_times) + ")");
                const std::string louds_io_suffix = is_lazy_mode
                                                        ? ("  (random-io+decode wall: " + std::to_string(
                                                            report_parent_louds_total_wall_sec) + "s)")
                                                        : ((report_preload_total_wall_sec > 0.0)
                                                               ? ("  (shared preload io+decode wall: " + std::to_string(
                                                                   report_preload_total_wall_sec) + "s)")
                                                               : std::string());
                LogInfo(
                    "Disk IVF linkage parent LOUDS decode CPU time " + louds_mode_note + ": " + std::to_string(
                        report_parent_louds_cpu_sec) + "s" + louds_io_suffix);
            }
            if (use_coeff_codec) {
                if (!bundle || !bundle->louds_timing_reported) {
                    const std::string huffman_mode_note = is_lazy_mode
                                                              ? ("(cpu-only bench wall, clusters=" + std::to_string(
                                                                  static_cast<int>(bundle
                                                                      ? bundle->lazy_loaded_cids.size()
                                                                      : lazy_coeff_decoded_count)) + ")")
                                                              : ("(preload all-cluster bench, clusters=" +
                                                                  std::to_string(
                                                                      bundle ? bundle->bench_n_clusters : 0) +
                                                                  ", bench_times=" + std::to_string(bench_times) + ")");
                    const std::string huffman_io_suffix = is_lazy_mode
                                                              ? ("  (random-io+decode wall: " + std::to_string(
                                                                  report_coeff_total_wall_sec) + "s)")
                                                              : ((report_preload_total_wall_sec > 0.0)
                                                                     ? ("  (shared preload io+decode wall: " +
                                                                         std::to_string(report_preload_total_wall_sec) +
                                                                         "s)")
                                                                     : std::string());
                    LogInfo(
                        "Disk IVF linkage Huffman decode CPU time " + huffman_mode_note + ": " + std::to_string(
                            report_coeff_cpu_sec) + "s" + huffman_io_suffix);
                }
            }
            if (bundle && !bundle->parent_cache_logged) {
                LogParentCacheMemoryOnce(bundle);
            }
            if (use_coeff_codec) {
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                const bool gpu_scan_active = cfg.eval.linkage_gpu_scan_enable && cfg.runtime.use_cuda &&
                    !eval::IsDiskNorm2ModeLut(cfg.eval.disk_norm2_mode);
                (void)gpu_scan_active;
                LogInfo("Disk IVF linkage scan+topk time: " + std::to_string(total_core_scan_topk_wall) + "s");
#else
            LogInfo("Disk IVF linkage scan+topk time: " + std::to_string(total_core_scan_topk_wall) + "s");
#endif
            }
            else {
                LogInfo("Disk IVF linkage scan+topk time: " + std::to_string(total_core_scan_topk_wall) + "s");
            }
            LogInfo(
                "Disk IVF linkage core time (qt_rotate+qt_gemm+probe_sel+scan+topk): " + std::to_string(core_wall_sec) +
                "s");
            if (core_wall_sec > 0.0) {
                const int nq = query_dataset_inmem.Xq.cols;
                if (nq > 0) {
                    LogInfo("Disk IVF linkage QPS (core): " + std::to_string(static_cast<double>(nq) / core_wall_sec));
                }
            }
        }
        // Mark LOUDS/Huffman timing as reported so repeat runs don't duplicate the lines.
        if (bundle) bundle->louds_timing_reported = true;

        // ---- Norm2 cache: store to disk if configured and file does not yet exist ----
        {
            const bool want_store = use_coeff_codec
                                        ? cfg.eval.linkage_norm2_store_int8
                                        : cfg.eval.linkage_norm2_store_float;
            if (want_store) {
                if (eval::IsDiskNorm2ModeLut(cfg.eval.disk_norm2_mode)) {
                    if (eval::IsDiskNorm2ModeLutCluster(cfg.eval.disk_norm2_mode)) {
                        std::unordered_map<int, eval::Norm2Lut> ignored;
                        bool have_valid_cache =
                        (eval::LoadNorm2LutClusterCache(linkage_list, use_coeff_codec,
                                                        cfg.base.encode.hnorms,
                                                        cfg.eval.disk_norm2_lut_kmeans_niter,
                                                        &ignored, nullptr) ==
                            eval::Norm2LutDiskLoadResult::kLoaded);
                        if (!have_valid_cache) {
                            const auto lut_cache = provider.ExportNorm2LutCache();
                            if (!lut_cache.empty()) {
                                std::string store_err;
                                if (!eval::StoreNorm2LutClusterCache(linkage_list, use_coeff_codec,
                                                                     cfg.base.encode.hnorms,
                                                                     cfg.eval.disk_norm2_lut_kmeans_niter,
                                                                     lut_cache, &store_err)) {
                                    LogInfo("Skipping norm2 LUT cache store: " + store_err);
                                }
                            }
                        }
                    }
                }
                else {
                    const auto out_path = std::filesystem::path(linkage_list.dir()) / norm2_filename;
                    std::error_code fec;
                    bool have_valid_cache = false;
                    if (std::filesystem::exists(out_path, fec) && !fec) {
                        have_valid_cache = ValidateNorm2CacheHash(linkage_list, use_coeff_codec, out_path, nullptr);
                    }
                    if (!have_valid_cache) {
                        const auto norm_cache = provider.ExportNorm2Cache();
                        if (!norm_cache.empty()) {
                            std::string store_err;
                            if (!StoreNorm2CacheMap(linkage_list, use_coeff_codec, out_path, norm_cache, &store_err)) {
                                LogInfo("Skipping norm2 cache store for " + norm2_filename +
                                    ": " + store_err);
                            }
                        }
                    }
                }
            }
        }

        return true;
    }

    bool EvaluateRecallLinkageIvfFromDiskTimed(const Config& cfg,
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
        return EvalDiskLinkageRecallLookupImpl(
            cfg, query_dataset_inmem, linkage_list, train, use_coeff_codec, out, timing, session, err);
    }

    bool PrepareRecallLinkageIvfDiskSession(const Config& cfg,
                                            const io::LinkageListReader& linkage_list,
                                            const TrainResult& train,
                                            bool use_coeff_codec,
                                            DiskLinkageEvalTiming* timing,
                                            DiskLinkageEvalSession* session,
                                            std::string* err) {
        if (cfg.runtime.omp_threads > 0) {
            omp_set_num_threads(std::max(1, cfg.runtime.omp_threads));
        }
        const bool profile_timing = cfg.large.profile_timing;
        const int m = cfg.model.m;
        const int m_codes = std::max(0, m - 1);
        if (train.C_root.books.empty()) {
            if (err) *err = "PrepareRecallLinkageIvfDiskSession: empty root codebooks.";
            return false;
        }
        const ColMajorMatrix<float>& C_root0 = train.C_root.books.front();
        if (C_root0.rows <= 0 || m <= 1 || m_codes != linkage_list.m_codes()) {
            if (err) *err = "PrepareRecallLinkageIvfDiskSession: invalid dims / m_codes mismatch.";
            return false;
        }
        std::vector<const ColMajorMatrix<float>*> root_small_books;
        root_small_books.reserve(static_cast<std::size_t>(m_codes));
        for (int l = 1; l < m; ++l) {
            root_small_books.push_back(&train.C_root.books[static_cast<std::size_t>(l)]);
        }
        const CodebookMeta meta_root_small = BuildCodebookMeta(root_small_books);
        const CodebookMeta meta_one = BuildCodebookMeta(GatherBooks(train.C_one));
        const bool want_store = use_coeff_codec
                                    ? cfg.eval.linkage_norm2_store_int8
                                    : cfg.eval.linkage_norm2_store_float;
        const bool use_norm2_lut = eval::IsDiskNorm2ModeLut(cfg.eval.disk_norm2_mode);
        const bool use_global_lut = eval::IsDiskNorm2ModeLutGlobal(cfg.eval.disk_norm2_mode);
        const std::string norm2_filename = use_coeff_codec ? "norm2_int8.f32" : "norm2_float.f32";
        if (want_store) {
            if (use_norm2_lut) {
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
                    }
                    else {
                        LogInfo(std::string("Precomputing full ") +
                            (use_global_lut ? "global" : "cluster") +
                            " norm2 LUT cache ...");
                    }
                    if (!BuildAndStoreFullNorm2LutCache(cfg, linkage_list, C_root0, meta_root_small,
                                                        meta_one, use_coeff_codec, err)) {
                        return false;
                    }
                }
            }
            else {
                const auto norm2_path = std::filesystem::path(linkage_list.dir()) / norm2_filename;
                if (!ValidateNorm2CacheHash(linkage_list, use_coeff_codec, norm2_path, nullptr)) {
                    LogInfo("Precomputing full norm2 cache for " + norm2_filename + " ...");
                    if (!BuildAndStoreFullNorm2Cache(cfg, linkage_list, C_root0, meta_root_small,
                                                     meta_one, use_coeff_codec, norm2_path, err)) {
                        return false;
                    }
                }
            }
        }
        std::shared_ptr<ProviderBundle> bundle;
        if (!EnsureDiskEvalProviderBundle(cfg, linkage_list, C_root0, meta_root_small,
                                          meta_one, norm2_filename, use_coeff_codec,
                                          profile_timing, timing, session, &bundle, err)) {
            return false;
        }
        if (!PreparePreloadDecodeBenchmark(cfg, linkage_list, &bundle, profile_timing, err)) {
            return false;
        }

        if (cfg.eval.linkage_preload_clusters_io_threads >= 0 && bundle && bundle->bench_done &&
            !cfg.eval.bench_quiet && !bundle->louds_timing_reported) {
            const int bench_times = std::max(1, cfg.eval.linkage_louds_huffman_bench_times);
            if (cfg.eval.linkage_parent_louds_enable && linkage_list.meta().store_parent_louds != 0) {
                const std::string wall_suffix = (bundle->preload_total_wall_sec > 0.0)
                                                    ? ("  (shared preload io+decode wall: " + std::to_string(
                                                        bundle->preload_total_wall_sec) + "s)")
                                                    : std::string();
                LogInfo("Disk IVF linkage parent LOUDS decode CPU time (preload all-cluster bench, clusters=" +
                    std::to_string(bundle->bench_n_clusters) + ", bench_times=" +
                    std::to_string(bench_times) + "): " + std::to_string(bundle->bench_louds_wall_avg_sec) + "s" +
                    wall_suffix);
            }
            if (use_coeff_codec) {
                const std::string wall_suffix = (bundle->preload_total_wall_sec > 0.0)
                                                    ? ("  (shared preload io+decode wall: " + std::to_string(
                                                        bundle->preload_total_wall_sec) + "s)")
                                                    : std::string();
                LogInfo("Disk IVF linkage Huffman decode CPU time (preload all-cluster bench, clusters=" +
                    std::to_string(bundle->bench_n_clusters) + ", bench_times=" +
                    std::to_string(bench_times) + "): " + std::to_string(bundle->bench_coeff_wall_avg_sec) + "s" +
                    wall_suffix);
            }
            bundle->louds_timing_reported = true;
        }
        return true;
    }
} // namespace stlq
