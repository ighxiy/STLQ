#include "stlq/eval/linkage_cluster_provider.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <thread>

#include "stlq/coeff/huffman_canonical.h"
#include "stlq/coeff/span.h"
#include "stlq/core/threading.h"
#include "stlq/succinct/parent_louds.h"
#include "stlq/common/timer.h"
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
#include "stlq/eval/linkage_norm_provider_cuda.h"
#endif

#include <omp.h>

namespace stlq::eval {

namespace {

inline std::int8_t QFromSym(std::uint16_t sym, int base) {
    return std::int8_t(int(sym) - base);
}

void DecodeAllLayersIntoRanges(int m,
                                      int nc,
                                      std::int8_t* q_layer_major_out,  // size m*nc
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
            const std::uint16_t sym = model.DecodeSymbol(br);
            ql[i] = QFromSym(sym, base);
        }
        for (int i = r1_begin; i < r1_end; ++i) {
            const std::uint16_t sym = model.DecodeSymbol(br);
            ql[i] = QFromSym(sym, base);
        }
    }
}

void DecodeOneLayerIntoRanges(int nc,
                                     std::int8_t* q_layer_out,  // size nc
                                     int base,
                                     const HuffmanCanonicalModel& model,
                                     const std::vector<std::uint8_t>& payload,
                                     int r0_begin,
                                     int r0_end,
                                     int r1_begin,
                                     int r1_end) {
    (void)nc;
    BitReader br(payload.data(), payload.size());
    for (int i = r0_begin; i < r0_end; ++i) {
        const std::uint16_t sym = model.DecodeSymbol(br);
        q_layer_out[i] = QFromSym(sym, base);
    }
    for (int i = r1_begin; i < r1_end; ++i) {
        const std::uint16_t sym = model.DecodeSymbol(br);
        q_layer_out[i] = QFromSym(sym, base);
    }
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
    if (r0_begin >= r0_end && r1_begin >= r1_end) {
        return;
    }
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

}  // namespace

bool ClusterProvider::Open(const io::LinkageListReader& linkage_list,
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
                           std::string* err) {
    linkage_list_ = &linkage_list;
    coeff_reader_ = coeff_codec;
    norm_provider_ = norm_provider;
    use_coeff_codec_ = use_coeff_codec;
    prefer_parent_louds_ = prefer_parent_louds;
    parent_louds_select_stride_ =
        std::max<std::uint32_t>(1u, parent_louds_select_stride);
    parent_louds_rank_words_per_super_log2_ =
        std::max<std::uint32_t>(1u, std::min<std::uint32_t>(10u, parent_louds_rank_words_per_super_log2));
    parent_louds_build_indices_ = parent_louds_build_indices;
    adaptive_parent_u16_storage_ = adaptive_parent_u16_storage;
    use_norm2_lut_ = use_norm2_lut;
    use_norm2_lut_global_ = use_norm2_lut && use_norm2_lut_global;
    norm2_lut_h_ = std::max(1, norm2_lut_h);
    norm2_lut_kmeans_niter_ = std::max(1, norm2_lut_kmeans_niter);
    norm2_disk_lazy_enabled_ = false;
    profile_prep_stats_ = profile_prep_stats;
    global_norm2_lut_centers_.clear();

    list_cache_.clear();
    coeff_cache_.clear();
    norm_cache_.clear();
    preloaded_ = false;

    if (!linkage_thr_.OpenFrom(linkage_list,
                             /*require_coeffs_f32=*/!use_coeff_codec,
                             err)) {
        return false;
    }
    if (linkage_list.small_code_width_bytes() != 1 || linkage_list.code0_width_bytes() != 1) {
        if (err) *err = "ClusterProvider: eval only supports uint8 linkage_list codes.";
        return false;
    }
    if (use_coeff_codec_) {
        if (!coeff_reader_) {
            if (err) *err = "ClusterProvider: coeff codec requested but reader is null.";
            return false;
        }
        if (!coeff_thr_.OpenFrom(*coeff_reader_, err)) {
            return false;
        }
    }
    return true;
}

void ClusterProvider::MaybeCompactParent(LinkageListEntry* out) {
    if (!out || !adaptive_parent_u16_storage_ || out->parent.empty() || out->nc <= 0 ||
        out->nc > static_cast<int>(std::numeric_limits<std::uint16_t>::max())) {
        return;
    }
    out->parent_u16.resize(out->parent.size());
    for (std::size_t i = 0; i < out->parent.size(); ++i) {
        out->parent_u16[i] = static_cast<std::uint16_t>(out->parent[i]);
    }
    out->parent.clear();
    out->parent.shrink_to_fit();
    out->parent_stored_as_u16 = true;
}

bool ClusterProvider::LoadLinkageListEntry(int cid, LinkageListEntry* out, std::string* err) {
    return LoadLinkageListEntryWith(linkage_thr_, cid, out, profile_prep_stats_, err);
}

bool ClusterProvider::LoadLinkageListEntryWith(io::LinkageListThreadReader& thr,
                                              int cid, LinkageListEntry* out,
                                              bool profile, std::string* err) {
    if (!out) {
        return false;
    }
    out->cid = cid;
    out->parent_louds_decode_sec = 0.0;
    out->parent_louds_read_sec = 0.0;
    out->parent_louds_deser_sec = 0.0;
    out->parent_louds_decode_parent_sec = 0.0;
    out->parent_louds_slice_sec = 0.0;
    std::string ignored;
    const bool need_f32 = !use_coeff_codec_;

    std::uint32_t span_n_real = 0;
    std::uint32_t span_n_virt = 0;
    if (linkage_list_) {
        std::uint64_t real_lo = 0, real_hi = 0, virt_lo = 0, virt_hi = 0, depth_lo = 0, depth_hi = 0;
        std::string span_err;
        if (linkage_list_->ReadClusterSpan(cid, &real_lo, &real_hi, &virt_lo, &virt_hi, &depth_lo, &depth_hi, &span_err)) {
            const std::uint64_t n_real64 = real_hi - real_lo;
            const std::uint64_t n_virt64 = virt_hi - virt_lo;
            if (n_real64 <= static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) &&
                n_virt64 <= static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
                span_n_real = static_cast<std::uint32_t>(n_real64);
                span_n_virt = static_cast<std::uint32_t>(n_virt64);
            }
        }
    }

    bool want_parent_from_louds =
        prefer_parent_louds_ && (thr.meta().store_parent_louds != 0) && (span_n_real > 0);
    if (want_parent_from_louds) {
        std::vector<std::uint8_t> blob;
        std::string louds_err;
        struct LoudsBreakdown {
            double read_sec = 0.0;
            double deser_sec = 0.0;
            double decode_parent_sec = 0.0;
            double slice_sec = 0.0;
            [[nodiscard]] double Total() const { return read_sec + deser_sec + decode_parent_sec + slice_sec; }
        };
        const auto decode_parent_louds = [&](LoudsBreakdown* out_breakdown) -> bool {
            if (out_breakdown) {
                *out_breakdown = LoudsBreakdown{};
            }

            if (out_breakdown) {
                Timer t;
                if (!thr.ReadClusterParentLOUDSBlob(cid, &blob, &louds_err) || blob.empty()) {
                    return false;
                }
                out_breakdown->read_sec = t.ElapsedSeconds();
            } else {
                if (!thr.ReadClusterParentLOUDSBlob(cid, &blob, &louds_err) || blob.empty()) {
                    return false;
                }
            }

            try {
                succinct::ParentLOUDS louds;
                if (out_breakdown) {
                    Timer t;
                    louds.Deserialize(blob.data(),
                                      blob.size(),
                                      parent_louds_rank_words_per_super_log2_,
                                      parent_louds_build_indices_,
                                      parent_louds_select_stride_);
                    out_breakdown->deser_sec = t.ElapsedSeconds();
                } else {
                    louds.Deserialize(blob.data(),
                                      blob.size(),
                                      parent_louds_rank_words_per_super_log2_,
                                      parent_louds_build_indices_,
                                      parent_louds_select_stride_);
                }

                const auto n_real = static_cast<std::size_t>(span_n_real);
                const auto n_virt = static_cast<std::size_t>(span_n_virt);
                const std::size_t nc = n_real + n_virt;
                std::vector<std::uint32_t> parent_all;

                if (out_breakdown) {
                    Timer t;
                    louds.DecodeParent1Based(&parent_all, nc);
                    out_breakdown->decode_parent_sec = t.ElapsedSeconds();
                } else {
                    louds.DecodeParent1Based(&parent_all, nc);
                }

                if (out_breakdown) {
                    Timer t;
                    out->parent.resize(n_real);
                    for (std::size_t i = 0; i < n_real; ++i) {
                        out->parent[i] = parent_all[n_virt + i];
                    }
                    out_breakdown->slice_sec = t.ElapsedSeconds();
                } else {
                    out->parent.resize(n_real);
                    for (std::size_t i = 0; i < n_real; ++i) {
                        out->parent[i] = parent_all[n_virt + i];
                    }
                }

                return true;
            } catch (...) {
                out->parent.clear();
                return false;
            }
        };

        if (profile) {
            LoudsBreakdown bd;
            want_parent_from_louds = decode_parent_louds(&bd);
            out->parent_louds_read_sec = bd.read_sec;
            out->parent_louds_deser_sec = bd.deser_sec;
            out->parent_louds_decode_parent_sec = bd.decode_parent_sec;
            out->parent_louds_slice_sec = bd.slice_sec;
            // CPU-only decode time (no I/O): deser + decode_parent + slice.
            out->parent_louds_decode_sec = bd.deser_sec + bd.decode_parent_sec + bd.slice_sec;
        } else {
            // Even without fine-grained profiling we need both I/O and CPU-only decode time.
            LoudsBreakdown bd;
            want_parent_from_louds = decode_parent_louds(&bd);
            // CPU-only decode time (deser + decode + slice); I/O stored separately in read_sec.
            out->parent_louds_decode_sec = bd.deser_sec + bd.decode_parent_sec + bd.slice_sec;
            out->parent_louds_read_sec   = bd.read_sec;
        }
    }

    if (!thr.ReadCluster(cid,
                         &out->real_ids,
                         want_parent_from_louds ? nullptr : &out->parent,
                         &out->depth_offsets,
                         &out->codes_small,
                         need_f32 ? &out->coeffs_small_f32 : nullptr,
                         &out->code0_one,
                         need_f32 ? &out->a0 : nullptr,
                         &out->virt_codes_small,
                         need_f32 ? &out->virt_coeffs_small_f32 : nullptr,
                         need_f32 ? &out->virt_a0 : nullptr,
                         &ignored)) {
        if (err && err->empty()) {
            *err = "ClusterProvider: failed to read linkage_list cluster cid=" + std::to_string(cid);
        }
        return false;
    }

    out->n_real = static_cast<int>(out->real_ids.size());
    // Empty cluster is valid (can happen when nlist is large or in edge datasets).
    // For recall, an empty cluster contributes nothing and should not abort evaluation.
    if (out->n_real == 0) {
        out->n_virt = 0;
        out->nc = 0;
        out->n_root_real = 0;
        out->depth_offsets.assign({0u, 0u});
        return true;
    }
    const int parent_size = out->parent_stored_as_u16
        ? static_cast<int>(out->parent_u16.size())
        : static_cast<int>(out->parent.size());
    if (out->n_real < 0 || parent_size != out->n_real) {
        if (err) *err = "ClusterProvider: invalid linkage_list cluster sizes cid=" + std::to_string(cid);
        return false;
    }

    const int m_codes = thr.m_codes();
    // Eval contract: small-layer codes are stored as uint8 payloads.
    out->n_virt = (m_codes > 0) ? static_cast<int>(out->virt_codes_small.size()) / m_codes : 0;
    out->nc = out->n_real + out->n_virt;

    if (out->depth_offsets.size() < 2) {
        int roots = 0;
        for (int i = 0; i < out->n_real; ++i) {
            const std::uint32_t p = out->parent_stored_as_u16
                ? static_cast<std::uint32_t>(out->parent_u16[static_cast<std::size_t>(i)])
                : out->parent[static_cast<std::size_t>(i)];
            if (p != 0) {
                break;
            }
            ++roots;
        }
        out->depth_offsets.assign({0u, static_cast<std::uint32_t>(roots),
                                   static_cast<std::uint32_t>(out->n_real)});
        out->n_root_real = roots;
    } else {
        out->n_root_real = static_cast<int>(out->depth_offsets[1]);
    }
    MaybeCompactParent(out);
    return true;
}

bool ClusterProvider::LoadAndDecodeCoeff(int cid,
                                         const LinkageListEntry& list,
                                         CoeffEntry* out,
                                         std::string* err) {
    return LoadAndDecodeCoeffWith(coeff_thr_, cid, list, out, err);
}

bool ClusterProvider::LoadAndDecodeCoeffWith(io::LinkageCoeffCodecThreadReader& cthr,
                                              int cid,
                                              const LinkageListEntry& list,
                                              CoeffEntry* out,
                                              std::string* err) {
    if (!out || !coeff_reader_) {
        return false;
    }
    out->cid = cid;
    out->nc = list.nc;
    out->scales_root.clear();
    out->scales_linkage.clear();
    out->lens_root.clear();
    out->lens_linkage.clear();
    out->payload_root.clear();
    out->payload_linkage.clear();

    {
        Timer t_io;
        if (!cthr.ReadCluster(cid,
                              &out->scales_root,
                              &out->scales_linkage,
                              &out->lens_root,
                              &out->lens_linkage,
                              &out->payload_root,
                              &out->payload_linkage,
                              err)) {
            return false;
        }
        out->io_sec = t_io.ElapsedSeconds();
    }

    const int streams = coeff_reader_->meta().streams;
    const int S = coeff_reader_->meta().S;
    if (streams <= 0 || S <= 0) {
        if (err) *err = "ClusterProvider: invalid coeff codec meta.";
        return false;
    }
    const int base = (S == 256) ? 128 : (S / 2);

    out->q_layer_major.assign(static_cast<std::size_t>(linkage_list_->m_codes() + 1) *
                                  static_cast<std::size_t>(list.nc),
                              std::int8_t(0));
    // m is not stored in linkage_list meta; it is derived from model and must match scales sizes.
    const int m = static_cast<int>(out->scales_root.size());
    if (m <= 1 || static_cast<int>(out->scales_linkage.size()) != m) {
        if (err) *err = "ClusterProvider: invalid scale sizes for coeff codec.";
        return false;
    }
    if (static_cast<int>(out->q_layer_major.size()) != m * list.nc) {
        out->q_layer_major.assign(static_cast<std::size_t>(m) * static_cast<std::size_t>(list.nc),
                                  std::int8_t(0));
    }

    try {
        Timer t_decode;
        // Root group: virtual roots then real roots (virtual-front local id).
        DecodeGroupIntoRanges(m, list.nc, streams, S, base,
                              out->lens_root, out->payload_root,
                              0, list.n_virt,
                              list.n_virt, list.n_virt + list.n_root_real,
                              out->q_layer_major.data());
        // Linkage group: real linkage only.
        DecodeGroupIntoRanges(m, list.nc, streams, S, base,
                              out->lens_linkage, out->payload_linkage,
                              list.n_virt + list.n_root_real, list.nc,
                              0, 0,
                              out->q_layer_major.data());
        // Store CPU-only Huffman decode time (no I/O).
        out->cpu_decode_sec = t_decode.ElapsedSeconds();
    } catch (const std::exception& e) {
        if (err) {
            *err = std::string("ClusterProvider: coeff decode failed: ") + e.what();
        }
        return false;
    }

    return true;
}

void ClusterProvider::PreloadNorm2Cache(std::unordered_map<int, std::vector<float>>&& preloaded) {
    for (auto& [cid, v] : preloaded) {
        if (norm_cache_.find(cid) == norm_cache_.end()) {
            NormEntry entry;
            entry.cid = cid;
            entry.r_norm2 = std::move(v);
            norm_cache_.emplace(cid, std::move(entry));
        }
    }
}

void ClusterProvider::PreloadNorm2LutCache(std::unordered_map<int, Norm2Lut>&& preloaded) {
    for (auto& [cid, lut] : preloaded) {
        if (norm_cache_.find(cid) == norm_cache_.end()) {
            NormEntry entry;
            entry.cid = cid;
            entry.lut = std::move(lut);
            norm_cache_.emplace(cid, std::move(entry));
        }
    }
}

void ClusterProvider::PreloadGlobalNorm2LutCache(Norm2Lut&& preloaded) {
    global_norm2_lut_centers_ = std::move(preloaded.centers);
    if (!linkage_list_) return;
    const auto& real_offs = linkage_list_->real_offsets();
    const int nlist = linkage_list_->nlist();
    if (real_offs.size() != static_cast<std::size_t>(nlist + 1)) return;
    for (int cid = 0; cid < nlist; ++cid) {
        if (norm_cache_.find(cid) != norm_cache_.end()) continue;
        const auto lo = real_offs[static_cast<std::size_t>(cid)];
        const auto hi = real_offs[static_cast<std::size_t>(cid + 1)];
        const auto nr = hi - lo;
        if (nr == 0) continue;
        if (hi > preloaded.codes_u8.size()) break;
        NormEntry entry;
        entry.cid = cid;
        entry.lut.codes_u8.resize(static_cast<std::size_t>(nr));
        std::memcpy(entry.lut.codes_u8.data(),
                    preloaded.codes_u8.data() + static_cast<std::size_t>(lo),
                    static_cast<std::size_t>(nr));
        norm_cache_.emplace(cid, std::move(entry));
    }
}

void ClusterProvider::SetGlobalNorm2LutCenters(std::vector<float>&& centers) {
    global_norm2_lut_centers_ = std::move(centers);
}

void ClusterProvider::SetNorm2DiskLazyEnabled(bool enabled) {
    norm2_disk_lazy_enabled_ = enabled;
}

std::unordered_map<int, std::vector<float>> ClusterProvider::ExportNorm2Cache() const {
    std::unordered_map<int, std::vector<float>> out;
    out.reserve(norm_cache_.size());
    for (const auto& [cid, entry] : norm_cache_) {
        out.emplace(cid, entry.r_norm2);
    }
    return out;
}

std::unordered_map<int, Norm2Lut> ClusterProvider::ExportNorm2LutCache() const {
    std::unordered_map<int, Norm2Lut> out;
    out.reserve(norm_cache_.size());
    for (const auto& [cid, entry] : norm_cache_) {
        if (!entry.lut.codes_u8.empty() && !entry.lut.centers.empty()) {
            out.emplace(cid, entry.lut);
        }
    }
    return out;
}

bool ClusterProvider::PrecomputeAllNorm2(std::unordered_map<int, std::vector<float>>* out,
                                         int n_io_threads,
                                         std::string* err) {
    if (!out) return false;
    out->clear();
    if (!linkage_list_) return true;
    if (!norm_provider_) {
        if (err) *err = "PrecomputeAllNorm2: norm provider is null.";
        return false;
    }

    const int nlist = linkage_list_->nlist();
    if (nlist <= 0) return true;

    const int omp_default = GetOmpDefaultThreads();
    const bool provider_thread_safe = norm_provider_->IsThreadSafeForParallelPrecompute();
    const int requested_threads = std::max(
        1, std::min((n_io_threads == 0) ? std::max(1, omp_default) : std::max(1, n_io_threads), nlist));
    std::vector<std::unique_ptr<IClusterNormProvider>> provider_clones;
    std::vector<IClusterNormProvider*> thread_providers;
    int n_threads = requested_threads;
    if (provider_thread_safe) {
        thread_providers.assign(static_cast<std::size_t>(n_threads), norm_provider_);
    } else {
        provider_clones.reserve(static_cast<std::size_t>(n_threads));
        for (int t = 0; t < n_threads; ++t) {
            auto clone = norm_provider_->CloneForParallelPrecompute();
            if (!clone) {
                provider_clones.clear();
                n_threads = 1;
                break;
            }
            thread_providers.push_back(clone.get());
            provider_clones.push_back(std::move(clone));
        }
        if (n_threads == 1) {
            thread_providers.assign(1u, norm_provider_);
        }
    }

    std::vector<io::LinkageListThreadReader> thr_list(static_cast<std::size_t>(n_threads));
    for (int t = 0; t < n_threads; ++t) {
        std::string e;
        if (!thr_list[static_cast<std::size_t>(t)].OpenFrom(
                *linkage_list_, /*require_coeffs_f32=*/!use_coeff_codec_, &e)) {
            if (err) *err = "PrecomputeAllNorm2: list reader open failed: " + e;
            return false;
        }
    }

    std::vector<io::LinkageCoeffCodecThreadReader> thr_coeff(
        (use_coeff_codec_ && coeff_reader_) ? static_cast<std::size_t>(n_threads) : 0u);
    for (std::size_t t = 0; t < thr_coeff.size(); ++t) {
        std::string e;
        if (!thr_coeff[t].OpenFrom(*coeff_reader_, &e)) {
            if (err) *err = "PrecomputeAllNorm2: coeff reader open failed: " + e;
            return false;
        }
    }

    struct NormResult {
        int cid = -1;
        std::vector<float> r_norm2;
    };
    const auto per_thread = static_cast<std::size_t>((nlist + n_threads - 1) / n_threads);
    std::vector<std::vector<NormResult>> results(static_cast<std::size_t>(n_threads));
    for (auto& v : results) v.reserve(per_thread);
    std::vector<std::string> thread_errs(static_cast<std::size_t>(n_threads));
    std::vector<bool> thread_ok(static_cast<std::size_t>(n_threads), true);

    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(n_threads));
    for (int t = 0; t < n_threads; ++t) {
        threads.emplace_back([&, t]() {
            const std::size_t ti = static_cast<std::size_t>(t);
            for (int cid = t; cid < nlist; cid += n_threads) {
                LinkageListEntry list;
                std::string local_err;
                if (!LoadLinkageListEntryWith(thr_list[ti], cid, &list, /*profile=*/false, &local_err)) {
                    thread_ok[ti] = false;
                    thread_errs[ti] = local_err;
                    return;
                }
                if (list.n_real <= 0 || list.nc != list.n_real + list.n_virt) {
                    continue;
                }

                CoeffEntry coeff;
                CoeffEntry* coeff_ptr = nullptr;
                if (use_coeff_codec_) {
                    if (!LoadAndDecodeCoeffWith(thr_coeff[ti], cid, list, &coeff, &local_err)) {
                        thread_ok[ti] = false;
                        thread_errs[ti] = local_err;
                        return;
                    }
                    coeff_ptr = &coeff;
                }

                ClusterView cv;
                cv.cid = cid;
                cv.m_codes = linkage_list_->m_codes();
                cv.n_real = list.n_real;
                cv.n_virt = list.n_virt;
                cv.nc = list.nc;
                cv.n_root_real = list.n_root_real;
                cv.depth_offsets_len = static_cast<int>(list.depth_offsets.size());
                cv.real_ids = list.real_ids.data();
                cv.parent_is_u16 = list.parent_stored_as_u16;
                cv.parent_1based = list.parent_stored_as_u16 ? nullptr : list.parent.data();
                cv.parent_1based_u16 = list.parent_stored_as_u16 ? list.parent_u16.data() : nullptr;
                cv.depth_offsets = list.depth_offsets.data();
                cv.codes_small_bytes = list.codes_small.data();
                cv.code0_one_bytes = list.code0_one.data();
                cv.virt_codes_small_bytes = list.virt_codes_small.data();
                if (use_coeff_codec_) {
                    cv.m = static_cast<int>(coeff_ptr->scales_root.size());
                    cv.q_layer_major = coeff_ptr->q_layer_major.data();
                    cv.scales_root = coeff_ptr->scales_root.data();
                    cv.scales_linkage = coeff_ptr->scales_linkage.data();
                } else {
                    cv.m = cv.m_codes + 1;
                    cv.a0 = list.a0.data();
                    cv.coeffs_small = list.coeffs_small_f32.data();
                    cv.virt_a0 = list.virt_a0.data();
                    cv.virt_coeffs_small = list.virt_coeffs_small_f32.data();
                }

                NormResult nr;
                nr.cid = cid;
                if (!thread_providers[ti]->ComputeNorm2(cv, &nr.r_norm2, &local_err)) {
                    thread_ok[ti] = false;
                    thread_errs[ti] = local_err;
                    return;
                }
                results[ti].push_back(std::move(nr));
            }
        });
    }
    for (auto& th : threads) th.join();

    for (int t = 0; t < n_threads; ++t) {
        if (!thread_ok[static_cast<std::size_t>(t)]) {
            if (err) *err = "PrecomputeAllNorm2: thread " + std::to_string(t) +
                             " failed: " + thread_errs[static_cast<std::size_t>(t)];
            return false;
        }
    }

    out->reserve(static_cast<std::size_t>(nlist));
    for (auto& vec : results) {
        for (auto& r : vec) {
            out->emplace(r.cid, std::move(r.r_norm2));
        }
    }
    return true;
}

bool ClusterProvider::PreloadAllClusters(int n_io_threads, PreloadStats* stats, std::string* err) {
    if (n_io_threads < 0) {
        // Explicitly disabled.
        return true;
    }
    if (!linkage_list_) return true;
    const int nlist = linkage_list_->nlist();
    if (nlist <= 0) return true;

    // Determine number of threads.  0 = follow the process default OMP thread count.
    const int omp_default = GetOmpDefaultThreads();
    const int n_threads = std::max(1, std::min(
        (n_io_threads == 0) ? std::max(1, omp_default) : n_io_threads, nlist));

    // Pre-size hash maps to avoid rehashing under concurrent merge.
    list_cache_.reserve(static_cast<std::size_t>(nlist));
    if (use_coeff_codec_) {
        coeff_cache_.reserve(static_cast<std::size_t>(nlist));
    }

    // Build per-thread LinkageListThreadReader instances (each has own ifstream file handles).
    std::vector<io::LinkageListThreadReader> thr_list(static_cast<std::size_t>(n_threads));
    for (int t = 0; t < n_threads; ++t) {
        std::string e;
        if (!thr_list[static_cast<std::size_t>(t)].OpenFrom(
                *linkage_list_, /*require_coeffs_f32=*/!use_coeff_codec_, &e)) {
            if (err) *err = "PreloadAllClusters: list reader open failed: " + e;
            return false;
        }
    }

    // Build per-thread LinkageCoeffCodecThreadReader instances if needed.
    std::vector<io::LinkageCoeffCodecThreadReader> thr_coeff(
        (use_coeff_codec_ && coeff_reader_) ? static_cast<std::size_t>(n_threads) : 0u);
    for (std::size_t t = 0; t < thr_coeff.size(); ++t) {
        std::string e;
        if (!thr_coeff[t].OpenFrom(*coeff_reader_, &e)) {
            if (err) *err = "PreloadAllClusters: coeff reader open failed: " + e;
            return false;
        }
    }

    // Per-thread result storage (no lock needed during the load phase).
    struct ListResult  { int cid; LinkageListEntry entry; };
    struct CoeffResult { int cid; CoeffEntry entry; };
    const auto per_thread = static_cast<std::size_t>((nlist + n_threads - 1) / n_threads);
    std::vector<std::vector<ListResult>>  list_results(static_cast<std::size_t>(n_threads));
    std::vector<std::vector<CoeffResult>> coeff_results(static_cast<std::size_t>(n_threads));
    for (std::size_t t = 0; t < static_cast<std::size_t>(n_threads); ++t) {
        list_results[t].reserve(per_thread);
        if (use_coeff_codec_) coeff_results[t].reserve(per_thread);
    }

    std::vector<std::string> thread_errs(static_cast<std::size_t>(n_threads));
    std::vector<bool>        thread_ok  (static_cast<std::size_t>(n_threads), true);
    // Per-thread CPU decode time accumulators (no I/O time included).
    std::vector<double> thread_louds_cpu(static_cast<std::size_t>(n_threads), 0.0);
    std::vector<double> thread_coeff_cpu(static_cast<std::size_t>(n_threads), 0.0);

    // Spawn threads.  Each thread owns its reader and accumulates results locally.
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(n_threads));
    for (int t = 0; t < n_threads; ++t) {
        threads.emplace_back([this, t, n_threads, nlist, &thr_list, &thr_coeff,
                              &list_results, &coeff_results,
                              &thread_errs, &thread_ok,
                              &thread_louds_cpu, &thread_coeff_cpu]() {
            const std::size_t ti = static_cast<std::size_t>(t);
            auto& my_thr          = thr_list[ti];
            auto& my_list_results = list_results[ti];
            auto& my_err          = thread_errs[ti];
            double my_louds_cpu   = 0.0;
            double my_coeff_cpu   = 0.0;

            for (int cid = t; cid < nlist; cid += n_threads) {
                if (list_cache_.count(cid)) continue;  // already cached (read-only check, safe)

                LinkageListEntry lentry;
                // profile=false: fine-grained per-field breakdown not needed during preload,
                // but parent_louds_decode_sec is still populated (CPU-only deser+decode+slice).
                if (!LoadLinkageListEntryWith(my_thr, cid, &lentry, /*profile=*/false, &my_err)) {
                    thread_ok[ti] = false;
                    return;
                }
                my_louds_cpu += lentry.parent_louds_decode_sec;

                if (use_coeff_codec_ && !thr_coeff.empty()) {
                    CoeffEntry cent;
                    if (!LoadAndDecodeCoeffWith(thr_coeff[ti], cid, lentry, &cent, &my_err)) {
                        thread_ok[ti] = false;
                        return;
                    }
                    my_coeff_cpu += cent.cpu_decode_sec;
                    coeff_results[ti].push_back({cid, std::move(cent)});
                }

                my_list_results.push_back({cid, std::move(lentry)});
            }
            thread_louds_cpu[ti] = my_louds_cpu;
            thread_coeff_cpu[ti] = my_coeff_cpu;
        });
    }
    for (auto& th : threads) th.join();

    // Check per-thread errors.
    for (int t = 0; t < n_threads; ++t) {
        if (!thread_ok[static_cast<std::size_t>(t)]) {
            if (err) *err = "PreloadAllClusters: thread " + std::to_string(t) +
                             " failed: " + thread_errs[static_cast<std::size_t>(t)];
            return false;
        }
    }

    // Single-threaded merge: move per-thread results into shared caches.
    for (std::size_t t = 0; t < static_cast<std::size_t>(n_threads); ++t) {
        for (auto& r : list_results[t]) {
            list_cache_.emplace(r.cid, std::move(r.entry));
        }
        if (use_coeff_codec_) {
            for (auto& r : coeff_results[t]) {
                coeff_cache_.emplace(r.cid, std::move(r.entry));
            }
        }
    }

    // Aggregate CPU decode times and return in stats (no I/O time included).
    if (stats) {
        stats->louds_cpu_decode_sec = 0.0;
        stats->coeff_cpu_decode_sec = 0.0;
        for (std::size_t t = 0; t < static_cast<std::size_t>(n_threads); ++t) {
            stats->louds_cpu_decode_sec += thread_louds_cpu[t];
            stats->coeff_cpu_decode_sec += thread_coeff_cpu[t];
        }
    }

    preloaded_ = true;
    return true;
}

bool ClusterProvider::GetCluster(int cid, ClusterView* out, PrepStats* stats, std::string* err) {
    if (!out) {
        return false;
    }
    if (stats) {
        stats->coeff_decode_sec = 0.0;
        stats->coeff_io_sec = 0.0;
        stats->norm_prep_sec = 0.0;
        stats->parent_louds_decode_sec = 0.0;
        stats->parent_louds_read_sec = 0.0;
        stats->parent_louds_deser_sec = 0.0;
        stats->parent_louds_decode_parent_sec = 0.0;
        stats->parent_louds_slice_sec = 0.0;
    }
    if (!linkage_list_) {
        if (err) *err = "ClusterProvider: not open.";
        return false;
    }

    auto it_list = list_cache_.find(cid);
    if (it_list == list_cache_.end()) {
        LinkageListEntry entry;
        if (!LoadLinkageListEntry(cid, &entry, err)) {
            return false;
        }
        if (stats) {
            if (entry.parent_louds_decode_sec > 0.0) {
                stats->parent_louds_decode_sec += entry.parent_louds_decode_sec;
            }
            if (entry.parent_louds_read_sec > 0.0) {
                stats->parent_louds_read_sec += entry.parent_louds_read_sec;
            }
            if (entry.parent_louds_deser_sec > 0.0) {
                stats->parent_louds_deser_sec += entry.parent_louds_deser_sec;
            }
            if (entry.parent_louds_decode_parent_sec > 0.0) {
                stats->parent_louds_decode_parent_sec += entry.parent_louds_decode_parent_sec;
            }
            if (entry.parent_louds_slice_sec > 0.0) {
                stats->parent_louds_slice_sec += entry.parent_louds_slice_sec;
            }
        }
        it_list = list_cache_.emplace(cid, std::move(entry)).first;
    }
    LinkageListEntry& list = it_list->second;

    // Empty cluster: valid, but contributes nothing to scan.
    // IMPORTANT: keep a non-null `depth_offsets` pointer (scan kernel reads depth_offsets[1]).
    if (list.nc == 0 || list.n_real == 0) {
        out->cid = cid;
        out->m = linkage_thr_.m_codes() + 1;
        out->m_codes = linkage_thr_.m_codes();
        out->n_real = list.n_real;
        out->n_virt = list.n_virt;
        out->nc = list.nc;
        out->n_root_real = list.n_root_real;
        out->real_ids = list.real_ids.empty() ? nullptr : list.real_ids.data();
        out->parent_is_u16 = list.parent_stored_as_u16;
        out->parent_1based = list.parent_stored_as_u16 || list.parent.empty() ? nullptr : list.parent.data();
        out->parent_1based_u16 = list.parent_stored_as_u16 && !list.parent_u16.empty() ? list.parent_u16.data() : nullptr;
        out->depth_offsets = list.depth_offsets.empty() ? nullptr : list.depth_offsets.data();
        out->codes_small_bytes = list.codes_small.empty() ? nullptr : list.codes_small.data();
        out->code0_one_bytes = list.code0_one.empty() ? nullptr : list.code0_one.data();
        out->virt_codes_small_bytes = list.virt_codes_small.empty() ? nullptr : list.virt_codes_small.data();
        out->a0 = nullptr;
        out->coeffs_small = nullptr;
        out->virt_a0 = nullptr;
        out->virt_coeffs_small = nullptr;
        out->q_layer_major = nullptr;
        out->scales_root = nullptr;
        out->scales_linkage = nullptr;
        out->r_norm2 = nullptr;
        out->r_norm2_lut_u8 = nullptr;
        out->r_norm2_lut_centers = nullptr;
        out->r_norm2_lut_size = 0;
        return true;
    }

    CoeffEntry* coeff = nullptr;
    if (use_coeff_codec_) {
        auto it_coeff = coeff_cache_.find(cid);
        if (it_coeff == coeff_cache_.end()) {
            CoeffEntry entry;
            if (!LoadAndDecodeCoeff(cid, list, &entry, err)) {
                return false;
            }
            if (stats) {
                // entry.cpu_decode_sec is the Huffman-decode-only time (no I/O).
                stats->coeff_decode_sec += entry.cpu_decode_sec;
                stats->coeff_io_sec += entry.io_sec;
            }
            it_coeff = coeff_cache_.emplace(cid, std::move(entry)).first;
        }
        coeff = &it_coeff->second;

    }

    auto it_norm = norm_cache_.find(cid);
    if (it_norm == norm_cache_.end()) {
        if (!norm_provider_ && !norm2_disk_lazy_enabled_) {
            if (err) *err = "ClusterProvider: norm provider is null.";
            return false;
        }

        ClusterView cv;
        cv.cid = cid;
        cv.m_codes = linkage_thr_.m_codes();
        cv.n_real = list.n_real;
        cv.n_virt = list.n_virt;
        cv.nc = list.nc;
        cv.n_root_real = list.n_root_real;
        cv.depth_offsets_len = static_cast<int>(list.depth_offsets.size());
        cv.real_ids = list.real_ids.data();
        cv.parent_is_u16 = list.parent_stored_as_u16;
        cv.parent_1based = list.parent_stored_as_u16 ? nullptr : list.parent.data();
        cv.parent_1based_u16 = list.parent_stored_as_u16 ? list.parent_u16.data() : nullptr;
        cv.depth_offsets = list.depth_offsets.data();
        cv.codes_small_bytes = list.codes_small.data();
        cv.code0_one_bytes = list.code0_one.data();
        cv.virt_codes_small_bytes = list.virt_codes_small.data();

        if (use_coeff_codec_) {
            cv.m = static_cast<int>(coeff->scales_root.size());
            cv.q_layer_major = coeff->q_layer_major.data();
            cv.scales_root = coeff->scales_root.data();
            cv.scales_linkage = coeff->scales_linkage.data();
        } else {
            cv.m = cv.m_codes + 1;
            cv.a0 = list.a0.data();
            cv.coeffs_small = list.coeffs_small_f32.data();
            cv.virt_a0 = list.virt_a0.data();
            cv.virt_coeffs_small = list.virt_coeffs_small_f32.data();
        }

        NormEntry nentry;
        nentry.cid = cid;
        bool loaded_from_disk = false;
        Timer t;
        if (norm2_disk_lazy_enabled_) {
            if (use_norm2_lut_) {
                if (use_norm2_lut_global_) {
                    loaded_from_disk = LoadNorm2LutClusterCodesSlice(*linkage_list_,
                                                                     use_coeff_codec_,
                                                                     cid,
                                                                     &nentry.lut.codes_u8,
                                                                     err);
                } else {
                    loaded_from_disk = LoadNorm2LutClusterSlice(*linkage_list_,
                                                                use_coeff_codec_,
                                                                cid,
                                                                &nentry.lut,
                                                                err);
                }
            } else {
                loaded_from_disk = LoadFloatNorm2ClusterSlice(*linkage_list_,
                                                              use_coeff_codec_,
                                                              cid,
                                                              &nentry.r_norm2,
                                                              err);
            }
            if (!loaded_from_disk) {
                return false;
            }
        } else if (use_norm2_lut_) {
            if (use_norm2_lut_global_) {
                if (err) *err = "ClusterProvider: global norm2 LUT must be preloaded before GetCluster.";
                return false;
            }
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
            if (auto* cuda_norm = dynamic_cast<cuda::LinkageNormProviderLookupCuda*>(norm_provider_)) {
                if (!cuda_norm->ComputeNorm2Lut(cv,
                                               norm2_lut_h_,
                                               norm2_lut_kmeans_niter_,
                                               &nentry.lut,
                                               err)) {
                    return false;
                }
            } else
#endif
            if (!norm_provider_->ComputeNorm2(cv, &nentry.r_norm2, err)) {
                return false;
            }
            if (nentry.lut.codes_u8.empty() &&
                !BuildNorm2Lut(nentry.r_norm2.data(),
                               static_cast<int>(nentry.r_norm2.size()),
                               norm2_lut_h_,
                               norm2_lut_kmeans_niter_,
                               &nentry.lut,
                               err)) {
                return false;
            }
        } else if (stats) {
            if (!norm_provider_->ComputeNorm2(cv, &nentry.r_norm2, err)) {
                return false;
            }
        } else {
            if (!norm_provider_->ComputeNorm2(cv, &nentry.r_norm2, err)) {
                return false;
            }
        }
        if (stats) {
            stats->norm_prep_sec += t.ElapsedSeconds();
        }
        it_norm = norm_cache_.emplace(cid, std::move(nentry)).first;
    }
    const NormEntry& norm = it_norm->second;

    out->cid = cid;
    out->m = use_coeff_codec_ ? static_cast<int>(coeff->scales_root.size()) : (linkage_thr_.m_codes() + 1);
    out->m_codes = linkage_thr_.m_codes();
    out->n_real = list.n_real;
    out->n_virt = list.n_virt;
    out->nc = list.nc;
    out->n_root_real = list.n_root_real;
    out->depth_offsets_len = static_cast<int>(list.depth_offsets.size());

    out->real_ids = list.real_ids.data();
    out->parent_is_u16 = list.parent_stored_as_u16;
    out->parent_1based = list.parent_stored_as_u16 ? nullptr : list.parent.data();
    out->parent_1based_u16 = list.parent_stored_as_u16 ? list.parent_u16.data() : nullptr;
    out->depth_offsets = list.depth_offsets.data();
    out->codes_small_bytes = list.codes_small.data();
    out->code0_one_bytes = list.code0_one.data();
    out->virt_codes_small_bytes = list.virt_codes_small.data();

    if (use_coeff_codec_) {
        out->q_layer_major = coeff->q_layer_major.data();
        out->scales_root = coeff->scales_root.data();
        out->scales_linkage = coeff->scales_linkage.data();
    } else {
        out->a0 = list.a0.data();
        out->coeffs_small = list.coeffs_small_f32.data();
        out->virt_a0 = list.virt_a0.data();
        out->virt_coeffs_small = list.virt_coeffs_small_f32.data();
    }

    out->r_norm2 = norm.r_norm2.data();
    out->r_norm2_lut_u8 = norm.lut.codes_u8.empty() ? nullptr : norm.lut.codes_u8.data();
    if (use_norm2_lut_global_) {
        out->r_norm2_lut_centers = global_norm2_lut_centers_.empty() ? nullptr : global_norm2_lut_centers_.data();
        out->r_norm2_lut_size = static_cast<int>(global_norm2_lut_centers_.size());
    } else {
        out->r_norm2_lut_centers = norm.lut.centers.empty() ? nullptr : norm.lut.centers.data();
        out->r_norm2_lut_size = static_cast<int>(norm.lut.centers.size());
    }
    return true;
}

bool ClusterProvider::BenchmarkDecodeWall(const std::vector<int>& cids,
                                          int bench_times,
                                          bool request_profile_breakdown,
                                          BenchmarkResult* out,
                                          std::string* /*err*/) {
    if (!out) return false;
    *out = BenchmarkResult{};
    if (bench_times <= 0 || cids.empty()) return true;

    // ---- Pre-read LOUDS raw blobs (one I/O read per cluster, NOT timed) ----
    struct LoudsBlob {
        int cid;
        std::uint32_t n_real;
        std::uint32_t n_virt;
        std::vector<std::uint8_t> blob;
    };
    std::vector<LoudsBlob> louds_blobs;
    const bool has_louds = prefer_parent_louds_ && linkage_list_ &&
                           (linkage_thr_.meta().store_parent_louds != 0);
    if (has_louds) {
        louds_blobs.reserve(cids.size());
        for (int cid : cids) {
            auto it = list_cache_.find(cid);
            if (it == list_cache_.end()) continue;
            const auto& le = it->second;
            if (le.n_real <= 0) continue;
            LoudsBlob lb;
            lb.cid = cid;
            lb.n_real = static_cast<std::uint32_t>(le.n_real);
            lb.n_virt = static_cast<std::uint32_t>(le.n_virt);
            std::string e;
            if (!linkage_thr_.ReadClusterParentLOUDSBlob(cid, &lb.blob, &e) || lb.blob.empty()) {
                continue;
            }
            louds_blobs.push_back(std::move(lb));
        }
    }

    // ---- Collect Huffman payloads (already in coeff_cache_, no I/O) ----
    struct CoeffRef {
        int cid;
        int nc;
        int n_virt;
        int n_root_real;
        const CoeffEntry* entry;
    };
    std::vector<CoeffRef> coeff_refs;
    if (use_coeff_codec_ && coeff_reader_) {
        coeff_refs.reserve(cids.size());
        for (int cid : cids) {
            auto it_c = coeff_cache_.find(cid);
            auto it_l = list_cache_.find(cid);
            if (it_c == coeff_cache_.end() || it_l == list_cache_.end()) continue;
            const auto& le = it_l->second;
            if (le.nc <= 0) continue;
            coeff_refs.push_back({cid, le.nc, le.n_virt, le.n_root_real, &it_c->second});
        }
    }

    const int streams = (coeff_reader_) ? coeff_reader_->meta().streams : 0;
    const int S = (coeff_reader_) ? coeff_reader_->meta().S : 0;
    const int huff_base = (S == 256) ? 128 : (S / 2);
    const int m = linkage_list_ ? (linkage_list_->m_codes() + 1) : 0;

    // ---- Benchmark LOUDS decode (wall, OMP-parallel) ----
    if (!louds_blobs.empty()) {
        const int n_blobs = static_cast<int>(louds_blobs.size());
        std::size_t max_nc = 0;
        for (const auto& lb : louds_blobs) {
            max_nc = std::max(max_nc, static_cast<std::size_t>(lb.n_real) + static_cast<std::size_t>(lb.n_virt));
        }
        std::atomic<std::uint64_t> louds_sink{0};
        for (int iter = 0; iter < bench_times; ++iter) {
            const double t0 = omp_get_wtime();
            #pragma omp parallel default(none) shared(louds_blobs, louds_sink, max_nc) firstprivate(n_blobs)
            {
                std::vector<std::uint32_t> parent_all;
                parent_all.reserve(max_nc);
                #pragma omp for schedule(static)
                for (int i = 0; i < n_blobs; ++i) {
                    const auto& lb = louds_blobs[static_cast<std::size_t>(i)];
                    try {
                        succinct::ParentLOUDS louds;
                        louds.Deserialize(lb.blob.data(), lb.blob.size(),
                                          parent_louds_rank_words_per_super_log2_,
                                          parent_louds_build_indices_,
                                          parent_louds_select_stride_);
                        const std::size_t nc = static_cast<std::size_t>(lb.n_real) +
                                               static_cast<std::size_t>(lb.n_virt);
                        parent_all.clear();
                        louds.DecodeParent1Based(&parent_all, nc);
                        if (lb.n_real > 0) {
                            louds_sink.fetch_add(parent_all[static_cast<std::size_t>(lb.n_virt)], std::memory_order_relaxed);
                        }
                    } catch (...) {}
                }
            }
            out->louds_wall_avg_sec += (omp_get_wtime() - t0);
        }
        (void)louds_sink;
        out->louds_wall_avg_sec /= bench_times;
    }

    // ---- Benchmark Huffman decode (wall, OMP-parallel) ----
    if (!coeff_refs.empty() && m > 0 && streams > 0) {
        const int n_refs = static_cast<int>(coeff_refs.size());
        std::size_t max_qbuf = 0;
        for (const auto& cr : coeff_refs) {
            max_qbuf = std::max(max_qbuf, static_cast<std::size_t>(m) * static_cast<std::size_t>(cr.nc));
        }
        std::atomic<std::int64_t> coeff_sink{0};
        for (int iter = 0; iter < bench_times; ++iter) {
            const double t0 = omp_get_wtime();
            #pragma omp parallel default(none) shared(coeff_refs, coeff_sink, max_qbuf) firstprivate(n_refs, m, streams, S, huff_base)
            {
                std::vector<std::int8_t> q_buf;
                q_buf.resize(max_qbuf, std::int8_t(0));
                #pragma omp for schedule(static)
                for (int i = 0; i < n_refs; ++i) {
                    const auto& cr = coeff_refs[static_cast<std::size_t>(i)];
                    const auto* ce = cr.entry;
                    const std::size_t need = static_cast<std::size_t>(m) * static_cast<std::size_t>(cr.nc);
                    std::fill_n(q_buf.data(), need, std::int8_t(0));
                    try {
                        DecodeGroupIntoRanges(m, cr.nc, streams, S, huff_base,
                                              ce->lens_root, ce->payload_root,
                                              0, cr.n_virt,
                                              cr.n_virt, cr.n_virt + cr.n_root_real,
                                              q_buf.data());
                        DecodeGroupIntoRanges(m, cr.nc, streams, S, huff_base,
                                              ce->lens_linkage, ce->payload_linkage,
                                              cr.n_virt + cr.n_root_real, cr.nc,
                                              0, 0,
                                              q_buf.data());
                        if (cr.nc > 0) {
                            coeff_sink.fetch_add(static_cast<std::int64_t>(q_buf[0]), std::memory_order_relaxed);
                        }
                    } catch (...) {}
                }
            }
            out->coeff_wall_avg_sec += (omp_get_wtime() - t0);
        }
        (void)coeff_sink;
        out->coeff_wall_avg_sec /= bench_times;
    }

    // ---- Optional profile breakdown (single extra LOUDS pass) ----
    if (request_profile_breakdown && !louds_blobs.empty()) {
        const int n_blobs = static_cast<int>(louds_blobs.size());
        std::size_t max_nc = 0;
        for (const auto& lb : louds_blobs) {
            max_nc = std::max(max_nc, static_cast<std::size_t>(lb.n_real) + static_cast<std::size_t>(lb.n_virt));
        }
        // Accumulate per-thread then sum (avoiding contention).
        const int n_thr = omp_get_max_threads();
        std::vector<double> thr_read(static_cast<std::size_t>(n_thr), 0.0);
        std::vector<double> thr_deser(static_cast<std::size_t>(n_thr), 0.0);
        std::vector<double> thr_decode(static_cast<std::size_t>(n_thr), 0.0);
        std::vector<double> thr_slice(static_cast<std::size_t>(n_thr), 0.0);

        #pragma omp parallel default(none) \
            shared(louds_blobs, thr_read, thr_deser, thr_decode, thr_slice, max_nc) \
            firstprivate(n_blobs)
        {
            const int tid = omp_get_thread_num();
            double my_deser = 0.0, my_decode = 0.0, my_slice = 0.0;
            std::vector<std::uint32_t> parent_all;
            parent_all.reserve(max_nc);
            #pragma omp for schedule(static)
            for (int i = 0; i < n_blobs; ++i) {
                const auto& lb = louds_blobs[static_cast<std::size_t>(i)];
                try {
                    succinct::ParentLOUDS louds;
                    Timer td;
                    louds.Deserialize(lb.blob.data(), lb.blob.size(),
                                      parent_louds_rank_words_per_super_log2_,
                                      parent_louds_build_indices_,
                                      parent_louds_select_stride_);
                    my_deser += td.ElapsedSeconds();

                    const std::size_t nc = static_cast<std::size_t>(lb.n_real) +
                                           static_cast<std::size_t>(lb.n_virt);
                    parent_all.clear();
                    Timer tp;
                    louds.DecodeParent1Based(&parent_all, nc);
                    my_decode += tp.ElapsedSeconds();

                    Timer ts;
                    if (lb.n_real > 0) {
                        volatile std::uint32_t sink = parent_all[static_cast<std::size_t>(lb.n_virt)];
                        (void)sink;
                    }
                    my_slice += ts.ElapsedSeconds();
                } catch (...) {}
            }
            thr_deser[static_cast<std::size_t>(tid)] = my_deser;
            thr_decode[static_cast<std::size_t>(tid)] = my_decode;
            thr_slice[static_cast<std::size_t>(tid)] = my_slice;
        }
        for (int t = 0; t < n_thr; ++t) {
            out->profile_louds_read_sec += thr_read[static_cast<std::size_t>(t)];
            out->profile_louds_deser_sec += thr_deser[static_cast<std::size_t>(t)];
            out->profile_louds_decode_parent_sec += thr_decode[static_cast<std::size_t>(t)];
            out->profile_louds_slice_sec += thr_slice[static_cast<std::size_t>(t)];
        }
    }

    return true;
}

void ClusterProvider::ReleaseCoeffRawPayloads() {
    for (auto& [cid, entry] : coeff_cache_) {
        entry.lens_root.clear();
        entry.lens_linkage.clear();
        entry.payload_root.clear();
        entry.payload_linkage.clear();
        // shrink_to_fit to actually return memory to the allocator.
        entry.lens_root.shrink_to_fit();
        entry.lens_linkage.shrink_to_fit();
        entry.payload_root.shrink_to_fit();
        entry.payload_linkage.shrink_to_fit();
    }
}

ClusterProvider::ParentStorageStats ClusterProvider::GetParentStorageStats() const {
    ParentStorageStats stats{};
    for (const auto& [cid, entry] : list_cache_) {
        stats.baseline_parent_u32_bytes += static_cast<std::uint64_t>(entry.n_real) * sizeof(std::uint32_t);
        if (entry.parent_stored_as_u16) {
            stats.resident_parent_bytes += static_cast<std::uint64_t>(entry.parent_u16.size()) * sizeof(std::uint16_t);
            ++stats.clusters_u16;
        } else {
            stats.resident_parent_bytes += static_cast<std::uint64_t>(entry.parent.size()) * sizeof(std::uint32_t);
            ++stats.clusters_u32;
        }
    }
    return stats;
}

}  // namespace stlq::eval
