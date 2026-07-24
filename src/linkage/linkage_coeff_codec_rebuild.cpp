#include "stlq/linkage/linkage_streaming_builders.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <omp.h>

#include "stlq/coeff/coeff_cluster_codec.h"
#include "stlq/core/blas.h"
#include "stlq/core/threading.h"
#include "stlq/io/linkage_coeff_codec_store.h"
#include "stlq/io/linkage_list_store.h"
#include "stlq/linkage/linkage_streaming_support.h"

namespace stlq {

using linkage::ClusterWorkBuf;
using linkage::SolveSymPosdefCholeskyRetry;

    bool RebuildLinkageCoeffCodecFromLinkageListVirtual(const Config& cfg,
                                                        const TrainResult& train,
                                                        const io::LinkageListReader& linkage_list,
                                                        std::string* err) {
        if (!cfg.large.linkage_coeff_codec.enabled) {
            if (err) *err = "RebuildLinkageCoeffCodecFromLinkageListVirtual: large.linkage_coeff_codec.enabled=false";
            return false;
        }
        const int d = train.C_root.books.empty() ? 0 : train.C_root.books.front().rows;
        const int m = cfg.model.m;
        const int m_codes = std::max(0, m - 1);
        if (d <= 0 || m <= 0) {
            if (err) *err = "RebuildLinkageCoeffCodecFromLinkageListVirtual: invalid d/m.";
            return false;
        }
        if (linkage_list.nlist() <= 0 || linkage_list.m_codes() != m_codes) {
            if (err)
                *err =
                    "RebuildLinkageCoeffCodecFromLinkageListVirtual: linkage_list meta mismatch (nlist/m_codes).";
            return false;
        }
        if (linkage_list.small_code_width_bytes() != 1 || linkage_list.code0_width_bytes() != 1) {
            if (err)
                *err =
                    "RebuildLinkageCoeffCodecFromLinkageListVirtual: only uint8 linkage_list stores are supported.";
            return false;
        }

        io::LinkageListThreadReader linkage_thr;
        if (!linkage_thr.OpenFrom(linkage_list, err)) {
            return false;
        }

        // Open codec writer (truncate/rebuild in-place).
        io::LinkageCoeffCodecRandomWriter coeff_writer;
        io::LinkageCoeffCodecStoreConfig ccfg;
        ccfg.dir = linkage_list.dir();
        ccfg.nlist = linkage_list.nlist();
        ccfg.m = m;
        ccfg.granularity = cfg.large.linkage_coeff_codec.granularity;
        int max_bits = 2;
        for (int b : cfg.large.linkage_coeff_codec.bits_per_layer) {
            max_bits = std::max(max_bits, b);
        }
        ccfg.max_bits = max_bits;
        ccfg.alphabet_size = 0;
        if (!coeff_writer.Open(ccfg, err)) {
            return false;
        }
        const io::LinkageCoeffCodecMeta* coeff_meta = &coeff_writer.meta();

        std::vector<float> coeff_scales_ones(static_cast<std::size_t>(m), 1.0f);
        std::vector<std::vector<std::uint8_t>> coeff_lens_zeros(
            static_cast<std::size_t>(coeff_meta->streams),
            std::vector<std::uint8_t>(static_cast<std::size_t>(coeff_meta->S), 0));
        std::vector<std::vector<std::uint8_t>> coeff_payload_empty(static_cast<std::size_t>(coeff_meta->streams),
                                                                   std::vector<std::uint8_t>());

        struct CoeffCodecReconPrecomp
        {
            int d = 0;
            int m = 0;

            // Root pack.
            std::vector<int> h_root;
            std::vector<int> root_small_offsets; // size m, offsets for layers 1..m-1 into C_root_small
            int H_root_small = 0;
            ColMajorMatrix<float> C_root_small; // d×H_root_small
            ColMajorMatrix<float> G_root_small; // H_root_small×H_root_small
            std::vector<float> norm0_root; // ||C_root[0][:,cid]||^2

            // One pack.
            std::vector<int> h_one;
            std::vector<int> one_offsets; // size m, offsets into flattened C_one_all
            int H_one_total = 0;
            ColMajorMatrix<float> G_one_all; // H_one_total×H_one_total

            // 1-based norm2 tables for weighted quantile.
            std::vector<std::vector<float>> norm2_root_layers; // size m, each (h+1)
            std::vector<std::vector<float>> norm2_one_layers; // size m, each (h+1)
            std::vector<Span<const float>> norm2_root_spans;
            std::vector<Span<const float>> norm2_one_spans;
        };

        auto norm2_col = [&](const float* x) -> float
        {
            float s = 0.0f;
#pragma omp simd reduction(+:s)
            for (int i = 0; i < d; ++i) {
                s += x[i] * x[i];
            }
            return s;
        };

        auto coeff_pre = std::make_unique<CoeffCodecReconPrecomp>();
        coeff_pre->d = d;
        coeff_pre->m = m;

        // Root pack small layers concat + small-small gram.
        coeff_pre->h_root.resize(static_cast<std::size_t>(m), 0);
        for (int l = 0; l < m; ++l) {
            coeff_pre->h_root[static_cast<std::size_t>(l)] = train.C_root.books[static_cast<std::size_t>(l)].cols;
        }
        coeff_pre->root_small_offsets.assign(static_cast<std::size_t>(m), 0);
        int Hs = 0;
        for (int l = 1; l < m; ++l) {
            coeff_pre->root_small_offsets[static_cast<std::size_t>(l)] = Hs;
            Hs += coeff_pre->h_root[static_cast<std::size_t>(l)];
        }
        coeff_pre->H_root_small = Hs;
        coeff_pre->C_root_small = ColMajorMatrix<float>(d, Hs);
        {
            int col = 0;
            for (int l = 1; l < m; ++l) {
                const auto& book = train.C_root.books[static_cast<std::size_t>(l)];
                for (int c = 0; c < book.cols; ++c) {
                    std::memcpy(coeff_pre->C_root_small.Col(col + c), book.Col(c),
                                sizeof(float) * static_cast<std::size_t>(d));
                }
                col += book.cols;
            }
        }
        coeff_pre->G_root_small = ColMajorMatrix<float>(Hs, Hs);
        {
            ScopedBlasThreads blas_scope(std::max(1, omp_get_max_threads()));
            Gemm(true, false, 1.0f, coeff_pre->C_root_small, coeff_pre->C_root_small, 0.0f, &coeff_pre->G_root_small);
        }

        // Root norms for layer0 (by cluster id).
        const int h0 = coeff_pre->h_root[0];
        coeff_pre->norm0_root.assign(static_cast<std::size_t>(std::max(0, h0)), 0.0f);
        for (int cid = 0; cid < h0; ++cid) {
            coeff_pre->norm0_root[static_cast<std::size_t>(cid)] =
                norm2_col(train.C_root.books[0].Col(cid));
        }

        // One pack gram (all layers).
        coeff_pre->h_one.resize(static_cast<std::size_t>(m), 0);
        coeff_pre->one_offsets.assign(static_cast<std::size_t>(m), 0);
        int Ho = 0;
        for (int l = 0; l < m; ++l) {
            coeff_pre->one_offsets[static_cast<std::size_t>(l)] = Ho;
            coeff_pre->h_one[static_cast<std::size_t>(l)] = train.C_one.books[static_cast<std::size_t>(l)].cols;
            Ho += coeff_pre->h_one[static_cast<std::size_t>(l)];
        }
        coeff_pre->H_one_total = Ho;
        ColMajorMatrix<float> C_one_all(d, Ho);
        {
            int col = 0;
            for (int l = 0; l < m; ++l) {
                const auto& book = train.C_one.books[static_cast<std::size_t>(l)];
                for (int c = 0; c < book.cols; ++c) {
                    std::memcpy(C_one_all.Col(col + c), book.Col(c),
                                sizeof(float) * static_cast<std::size_t>(d));
                }
                col += book.cols;
            }
        }
        coeff_pre->G_one_all = ColMajorMatrix<float>(Ho, Ho);
        {
            ScopedBlasThreads blas_scope(std::max(1, omp_get_max_threads()));
            Gemm(true, false, 1.0f, C_one_all, C_one_all, 0.0f, &coeff_pre->G_one_all);
        }

        // 1-based norm2 tables.
        coeff_pre->norm2_root_layers.resize(static_cast<std::size_t>(m));
        coeff_pre->norm2_one_layers.resize(static_cast<std::size_t>(m));
        for (int l = 0; l < m; ++l) {
            const auto& br = train.C_root.books[static_cast<std::size_t>(l)];
            auto& nr = coeff_pre->norm2_root_layers[static_cast<std::size_t>(l)];
            nr.assign(static_cast<std::size_t>(br.cols) + 1u, 0.0f);
            for (int c = 0; c < br.cols; ++c) {
                nr[static_cast<std::size_t>(c + 1)] = norm2_col(br.Col(c));
            }
            const auto& bo = train.C_one.books[static_cast<std::size_t>(l)];
            auto& no = coeff_pre->norm2_one_layers[static_cast<std::size_t>(l)];
            no.assign(static_cast<std::size_t>(bo.cols) + 1u, 0.0f);
            for (int c = 0; c < bo.cols; ++c) {
                no[static_cast<std::size_t>(c + 1)] = norm2_col(bo.Col(c));
            }
        }
        coeff_pre->norm2_root_spans.clear();
        coeff_pre->norm2_one_spans.clear();
        coeff_pre->norm2_root_spans.reserve(static_cast<std::size_t>(m));
        coeff_pre->norm2_one_spans.reserve(static_cast<std::size_t>(m));
        for (int l = 0; l < m; ++l) {
            const auto& nr = coeff_pre->norm2_root_layers[static_cast<std::size_t>(l)];
            const auto& no = coeff_pre->norm2_one_layers[static_cast<std::size_t>(l)];
            coeff_pre->norm2_root_spans.emplace_back(nr.data(), nr.size());
            coeff_pre->norm2_one_spans.emplace_back(no.data(), no.size());
        }

        std::atomic<bool> ok{true};
        std::mutex err_mu;
        std::string first_err;

        auto fail = [&](const std::string& e)
        {
            ok.store(false, std::memory_order_relaxed);
            std::lock_guard<std::mutex> guard(err_mu);
            if (first_err.empty()) {
                first_err = e.empty() ? "RebuildLinkageCoeffCodecFromLinkageListVirtual: failed." : e;
            }
        };

        const int nlist = linkage_list.nlist();
        auto process_cluster = [&](int cid,
                                   io::LinkageListThreadReader& linkage_thr,
                                   ClusterWorkBuf& buf,
                                   std::string& local_err) -> bool
        {
            // Read float coeffs + codes from linkage_list.
            if (!linkage_thr.ReadCluster(cid,
                                         &buf.real_ids_depth_order,
                                         &buf.parent_u32,
                                         &buf.depth_offsets_u32,
                                         &buf.codes_small_depth,
                                         &buf.coeffs_small_depth,
                                         &buf.code0_one_depth,
                                         &buf.a0_depth,
                                         &buf.virt_codes_small,
                                         &buf.virt_coeffs_small,
                                         &buf.virt_a0,
                                         &local_err)) {
                fail(local_err);
                return false;
            }

            const int n_real_out = static_cast<int>(buf.real_ids_depth_order.size());
            const int n_virt_out = (m_codes > 0)
                                       ? static_cast<int>(buf.virt_codes_small.size()) / (m_codes * 1)
                                       : 0;
            const int nc = n_real_out + n_virt_out;

            if (nc <= 0) {
                // Empty cluster: still write deterministic empty codec record.
                buf.lens_root = coeff_lens_zeros;
                buf.lens_linkage = coeff_lens_zeros;
                buf.payload_root = coeff_payload_empty;
                buf.payload_linkage = coeff_payload_empty;
                if (!coeff_writer.WriteClusterAt(cid,
                                                 coeff_scales_ones,
                                                 coeff_scales_ones,
                                                 buf.lens_root,
                                                 buf.lens_linkage,
                                                 buf.payload_root,
                                                 buf.payload_linkage,
                                                 &local_err)) {
                    fail(local_err);
                    return false;
                }
                return true;
            }

            buf.a_layer_major.resize(static_cast<std::size_t>(m) * static_cast<std::size_t>(nc));
            buf.is_linkage.resize(static_cast<std::size_t>(nc));
            buf.q_tmp.resize(static_cast<std::size_t>(m) * static_cast<std::size_t>(nc));

            const int real_base = n_virt_out;
            for (int v = 0; v < n_virt_out; ++v) {
                buf.is_linkage[static_cast<std::size_t>(v)] = 0u;
                buf.a_layer_major[static_cast<std::size_t>(v)] = buf.virt_a0[static_cast<std::size_t>(v)];
                for (int l = 1; l < m; ++l) {
                    const std::size_t offc =
                        static_cast<std::size_t>(v) * static_cast<std::size_t>(m_codes) +
                        static_cast<std::size_t>(l - 1);
                    buf.a_layer_major[static_cast<std::size_t>(l) * static_cast<std::size_t>(nc) +
                            static_cast<std::size_t>(v)] =
                        buf.virt_coeffs_small[offc];
                }
            }
            for (int pos = 0; pos < n_real_out; ++pos) {
                const int local = real_base + pos;
                buf.is_linkage[static_cast<std::size_t>(local)] =
                    (buf.parent_u32[static_cast<std::size_t>(pos)] != 0) ? 1u : 0u;
                buf.a_layer_major[static_cast<std::size_t>(local)] = buf.a0_depth[static_cast<std::size_t>(pos)];
                for (int l = 1; l < m; ++l) {
                    const std::size_t offc =
                        static_cast<std::size_t>(pos) * static_cast<std::size_t>(m_codes) +
                        static_cast<std::size_t>(l - 1);
                    buf.a_layer_major[static_cast<std::size_t>(l) * static_cast<std::size_t>(nc) +
                            static_cast<std::size_t>(local)] =
                        buf.coeffs_small_depth[offc];
                }
            }

            CoeffQuantConfig qcfg;
            qcfg.bits_per_layer = cfg.large.linkage_coeff_codec.bits_per_layer;
            qcfg.p_first_candidates = cfg.large.linkage_coeff_codec.p_first_candidates;
            qcfg.p_rest_candidates = cfg.large.linkage_coeff_codec.p_rest_candidates;
            qcfg.use_weighted_quantile = cfg.large.linkage_coeff_codec.use_weighted_quantile;
            qcfg.allow_clip = cfg.large.linkage_coeff_codec.allow_clip;
            qcfg.fit_scale = cfg.large.linkage_coeff_codec.fit_scale;

            HuffmanConfig hcfg;
            hcfg.alphabet_size = 0;
            if (cfg.large.linkage_coeff_codec.granularity == "layer" ||
                cfg.large.linkage_coeff_codec.granularity == "per_layer") {
                hcfg.granularity = HuffmanGranularity::kPerLayer;
            }
            else {
                hcfg.granularity = HuffmanGranularity::kPerClusterAllLayers;
            }

            // Build 1-based code indices for weighted quantile and recon-aligned scale fitting.
            buf.B_code1_layer_major.resize(static_cast<std::size_t>(m) * static_cast<std::size_t>(nc));
            auto read_code0_one = [&](int pos) -> int
            {
                return static_cast<int>(buf.code0_one_depth[static_cast<std::size_t>(pos)]);
            };
            for (int pos = 0; pos < n_real_out; ++pos) {
                const int local = real_base + pos;
                const bool ilinkage = (buf.is_linkage[static_cast<std::size_t>(local)] != 0);
                const int code0 = ilinkage ? read_code0_one(pos) : cid;
                buf.B_code1_layer_major[static_cast<std::size_t>(local)] = static_cast<std::uint32_t>(code0 + 1);
            }
            for (int v = 0; v < n_virt_out; ++v) {
                buf.B_code1_layer_major[static_cast<std::size_t>(v)] = static_cast<std::uint32_t>(cid + 1);
            }
            for (int l = 1; l < m; ++l) {
                for (int pos = 0; pos < n_real_out; ++pos) {
                    const int local = real_base + pos;
                    const std::size_t offc =
                        static_cast<std::size_t>(pos) * static_cast<std::size_t>(m_codes) +
                        static_cast<std::size_t>(l - 1);
                    const int code = static_cast<int>(buf.codes_small_depth[offc]);
                    buf.B_code1_layer_major[static_cast<std::size_t>(l) * static_cast<std::size_t>(nc) +
                            static_cast<std::size_t>(local)] =
                        static_cast<std::uint32_t>(code + 1);
                }
                for (int v = 0; v < n_virt_out; ++v) {
                    const std::size_t offc =
                        static_cast<std::size_t>(v) * static_cast<std::size_t>(m_codes) +
                        static_cast<std::size_t>(l - 1);
                    const int code = static_cast<int>(buf.virt_codes_small[offc]);
                    buf.B_code1_layer_major[static_cast<std::size_t>(l) * static_cast<std::size_t>(nc) +
                            static_cast<std::size_t>(v)] =
                        static_cast<std::uint32_t>(code + 1);
                }
            }

            // Stage-1: per-layer quantization (optionally weighted) to produce q (int8).
            CoeffQuantGroupResult qres_root;
            CoeffQuantGroupResult qres_one;
            std::fill(buf.q_tmp.begin(), buf.q_tmp.end(), std::int8_t(0));
            try {
                const std::vector<Span<const float>> empty_norms;
                const auto& n2_root = qcfg.use_weighted_quantile ? coeff_pre->norm2_root_spans : empty_norms;
                const auto& n2_one = qcfg.use_weighted_quantile ? coeff_pre->norm2_one_spans : empty_norms;
                QuantizeClusterCoeffsSplitRootLinkage(
                    m,
                    nc,
                    Span<const float>(buf.a_layer_major.data(), buf.a_layer_major.size()),
                    Span<const std::uint32_t>(buf.B_code1_layer_major.data(), buf.B_code1_layer_major.size()),
                    Span<const std::uint8_t>(buf.is_linkage.data(), buf.is_linkage.size()),
                    n2_root,
                    n2_one,
                    qcfg,
                    Span<std::int8_t>(buf.q_tmp.data(), buf.q_tmp.size()),
                    &qres_root,
                    &qres_one);
            }
            catch (const std::exception& e) {
                fail(std::string("RebuildLinkageCoeffCodecFromLinkageListVirtual: coeff quantize failed: ") + e.what());
                return false;
            }

            std::vector<float> scale_root = qres_root.scales;
            std::vector<float> scale_one = qres_one.scales;

            // Stage-2/3/4: recon-aligned global scale fit + optional ICM refine of q.
            const int sweeps = std::max(0, cfg.large.linkage_coeff_codec.q_refine_sweeps);
            const int Lref = (cfg.large.linkage_coeff_codec.q_refine_max_layer <= 0)
                                 ? m
                                 : std::min(m, cfg.large.linkage_coeff_codec.q_refine_max_layer);
            const int step_limit = std::max(0, cfg.large.linkage_coeff_codec.q_refine_step_limit);

            std::vector<int> Qvec(static_cast<std::size_t>(m), 0);
            for (int l = 0; l < m; ++l) {
                const int bits = std::max(2, qcfg.bits_per_layer[static_cast<std::size_t>(l)]);
                Qvec[static_cast<std::size_t>(l)] = (1 << (bits - 1)) - 1;
            }

            // Compute g0s for this cid (dot between C_root[0][:,cid] and flattened C_root small layers).
            buf.g0s_root_small.resize(static_cast<std::size_t>(coeff_pre->H_root_small));
            const float* c0 = train.C_root.books[0].Col(cid);
            for (int j = 0; j < coeff_pre->H_root_small; ++j) {
                const float* cs = coeff_pre->C_root_small.Col(j);
                float s = 0.0f;
#pragma omp simd reduction(+:s)
                for (int r = 0; r < d; ++r) {
                    s += c0[r] * cs[r];
                }
                buf.g0s_root_small[static_cast<std::size_t>(j)] = s;
            }

            auto dot_root = [&](int l, int code_l0, int k, int code_k0) -> float
            {
                if (l == 0 && k == 0) {
                    return coeff_pre->norm0_root[static_cast<std::size_t>(cid)];
                }
                if (l == 0 && k > 0) {
                    const int fk = coeff_pre->root_small_offsets[static_cast<std::size_t>(k)] + code_k0;
                    return buf.g0s_root_small[static_cast<std::size_t>(fk)];
                }
                if (l > 0 && k == 0) {
                    const int fl = coeff_pre->root_small_offsets[static_cast<std::size_t>(l)] + code_l0;
                    return buf.g0s_root_small[static_cast<std::size_t>(fl)];
                }
                const int fl = coeff_pre->root_small_offsets[static_cast<std::size_t>(l)] + code_l0;
                const int fk = coeff_pre->root_small_offsets[static_cast<std::size_t>(k)] + code_k0;
                return coeff_pre->G_root_small(fl, fk);
            };
            auto dot_one = [&](int l, int code_l0, int k, int code_k0) -> float
            {
                const int fl = coeff_pre->one_offsets[static_cast<std::size_t>(l)] + code_l0;
                const int fk = coeff_pre->one_offsets[static_cast<std::size_t>(k)] + code_k0;
                return coeff_pre->G_one_all(fl, fk);
            };

            auto fit_scales = [&](const std::vector<int>& idx,
                                  const std::vector<float>& fallback,
                                  auto&& dot_fn) -> std::vector<float>
            {
                std::vector<double> A(static_cast<std::size_t>(m) * static_cast<std::size_t>(m), 0.0);
                std::vector<double> b(static_cast<std::size_t>(m), 0.0);
                std::vector<double> x(static_cast<std::size_t>(m), 0.0);
                std::vector<double> dot_t(static_cast<std::size_t>(m), 0.0);

                for (int ii : idx) {
                    int code[32];
                    int qv[32];
                    double av[32];
                    for (int l = 0; l < m; ++l) {
                        code[l] = static_cast<int>(buf.B_code1_layer_major[static_cast<std::size_t>(l) * static_cast<
                                std::size_t>(nc) +
                            static_cast<std::size_t>(ii)]) - 1;
                        qv[l] = static_cast<int>(buf.q_tmp[static_cast<std::size_t>(l) * static_cast<std::size_t>(nc) +
                            static_cast<std::size_t>(ii)]);
                        av[l] = static_cast<double>(buf.a_layer_major[static_cast<std::size_t>(l) * static_cast<
                                std::size_t>(nc) +
                            static_cast<std::size_t>(ii)]);
                    }
                    for (int l = 0; l < m; ++l) {
                        double acc = 0.0;
                        for (int k = 0; k < m; ++k) {
                            acc += av[k] * static_cast<double>(dot_fn(l, code[l], k, code[k]));
                        }
                        dot_t[static_cast<std::size_t>(l)] = acc;
                    }
                    for (int l = 0; l < m; ++l) {
                        const int ql = qv[l];
                        if (ql == 0) continue;
                        b[static_cast<std::size_t>(l)] += static_cast<double>(ql) * dot_t[static_cast<std::size_t>(l)];
                        for (int k = 0; k < m; ++k) {
                            const int qk = qv[k];
                            if (qk == 0) continue;
                            A[static_cast<std::size_t>(l) * static_cast<std::size_t>(m) + static_cast<std::size_t>(k)]
                                +=
                                static_cast<double>(ql) * static_cast<double>(qk) *
                                static_cast<double>(dot_fn(l, code[l], k, code[k]));
                        }
                    }
                }
                if (!SolveSymPosdefCholeskyRetry(m, A.data(), b.data(), x.data())) {
                    return fallback;
                }
                std::vector<float> out(static_cast<std::size_t>(m), 0.0f);
                for (int l = 0; l < m; ++l) {
                    out[static_cast<std::size_t>(l)] = static_cast<float>(x[static_cast<std::size_t>(l)]);
                }
                return out;
            };

            auto refine_q_icm = [&](const std::vector<int>& idx,
                                    const std::vector<float>& scales,
                                    auto&& dot_fn)
            {
                if (idx.empty() || sweeps <= 0) {
                    return;
                }
                std::vector<double> dot_t(static_cast<std::size_t>(m), 0.0);
                for (int ii : idx) {
                    int code[32];
                    int qv[32];
                    double av[32];
                    for (int l = 0; l < m; ++l) {
                        code[l] = static_cast<int>(buf.B_code1_layer_major[static_cast<std::size_t>(l) * static_cast<
                                std::size_t>(nc) +
                            static_cast<std::size_t>(ii)]) - 1;
                        qv[l] = static_cast<int>(buf.q_tmp[static_cast<std::size_t>(l) * static_cast<std::size_t>(nc) +
                            static_cast<std::size_t>(ii)]);
                        av[l] = static_cast<double>(buf.a_layer_major[static_cast<std::size_t>(l) * static_cast<
                                std::size_t>(nc) +
                            static_cast<std::size_t>(ii)]);
                    }
                    for (int l = 0; l < m; ++l) {
                        double acc = 0.0;
                        for (int k = 0; k < m; ++k) {
                            acc += av[k] * static_cast<double>(dot_fn(l, code[l], k, code[k]));
                        }
                        dot_t[static_cast<std::size_t>(l)] = acc;
                    }
                    for (int sw = 0; sw < sweeps; ++sw) {
                        for (int l = 0; l < Lref; ++l) {
                            const auto sl = static_cast<double>(scales[static_cast<std::size_t>(l)]);
                            if (sl == 0.0) {
                                qv[l] = 0;
                                continue;
                            }
                            const auto cc = static_cast<double>(dot_fn(l, code[l], l, code[l]));
                            if (cc == 0.0) {
                                qv[l] = 0;
                                continue;
                            }
                            double sum_other = 0.0;
                            for (int k = 0; k < m; ++k) {
                                if (k == l) continue;
                                const int qk = qv[k];
                                if (qk == 0) continue;
                                sum_other += static_cast<double>(scales[static_cast<std::size_t>(k)]) *
                                    static_cast<double>(qk) *
                                    static_cast<double>(dot_fn(l, code[l], k, code[k]));
                            }
                            const double beta = sl * (dot_t[static_cast<std::size_t>(l)] - sum_other);
                            const double alpha = (sl * sl) * cc;
                            if (alpha == 0.0) {
                                continue;
                            }
                            const int Q = Qvec[static_cast<std::size_t>(l)];
                            const int qcur = qv[l];
                            int qnew = qcur;
                            if (step_limit > 0) {
                                const int lo = std::max(-Q, qcur - step_limit);
                                const int hi = std::min(Q, qcur + step_limit);
                                double bestv = alpha * double(qcur) * double(qcur) - 2.0 * beta * double(qcur);
                                for (int cand = lo; cand <= hi; ++cand) {
                                    const double v = alpha * double(cand) * double(cand) - 2.0 * beta * double(cand);
                                    if (v < bestv) {
                                        bestv = v;
                                        qnew = cand;
                                    }
                                }
                            }
                            else {
                                const double qcont = beta / alpha;
                                qnew = static_cast<int>(std::llround(qcont));
                                qnew = std::min(Q, std::max(-Q, qnew));
                            }
                            qv[l] = qnew;
                        }
                    }
                    for (int l = 0; l < Lref; ++l) {
                        const int q = std::min(127, std::max(-128, qv[l]));
                        buf.q_tmp[static_cast<std::size_t>(l) * static_cast<std::size_t>(nc) +
                            static_cast<std::size_t>(ii)] = static_cast<std::int8_t>(q);
                    }
                }
            };

            std::vector<int> idx_root;
            std::vector<int> idx_linkage;
            idx_root.reserve(static_cast<std::size_t>(nc));
            idx_linkage.reserve(static_cast<std::size_t>(nc));
            for (int ii = 0; ii < nc; ++ii) {
                if (buf.is_linkage[static_cast<std::size_t>(ii)] == 0) {
                    idx_root.push_back(ii);
                }
                else {
                    idx_linkage.push_back(ii);
                }
            }

            if (qcfg.fit_scale) {
                scale_root = fit_scales(idx_root, scale_root, dot_root);
                scale_one = fit_scales(idx_linkage, scale_one, dot_one);
            }
            if (sweeps > 0) {
                refine_q_icm(idx_root, scale_root, dot_root);
                refine_q_icm(idx_linkage, scale_one, dot_one);
                if (qcfg.fit_scale) {
                    scale_root = fit_scales(idx_root, scale_root, dot_root);
                    scale_one = fit_scales(idx_linkage, scale_one, dot_one);
                }
            }

            ClusterCoeffCompressed comp;
            try {
                comp = CompressClusterCoeffsFromQ(
                    static_cast<std::uint32_t>(cid),
                    m,
                    nc,
                    Span<const std::int8_t>(buf.q_tmp.data(), buf.q_tmp.size()),
                    Span<const std::uint8_t>(buf.is_linkage.data(), buf.is_linkage.size()),
                    scale_root,
                    scale_one,
                    qcfg,
                    hcfg);
            }
            catch (const std::exception& e) {
                fail(std::string("RebuildLinkageCoeffCodecFromLinkageListVirtual: coeff codec failed: ") + e.what());
                return false;
            }

            const int streams = coeff_meta->streams;
            const int S = coeff_meta->S;
            buf.lens_root.assign(static_cast<std::size_t>(streams),
                                 std::vector<std::uint8_t>(static_cast<std::size_t>(S), 0));
            buf.lens_linkage.assign(static_cast<std::size_t>(streams),
                                    std::vector<std::uint8_t>(static_cast<std::size_t>(S), 0));
            buf.payload_root.assign(static_cast<std::size_t>(streams), {});
            buf.payload_linkage.assign(static_cast<std::size_t>(streams), {});

            auto fill_group = [&](const ClusterGroupCompressed& g,
                                  std::vector<std::vector<std::uint8_t>>* lens_out,
                                  std::vector<std::vector<std::uint8_t>>* payload_out)
            {
                if (g.models.empty() || g.payloads.empty()) {
                    return;
                }
                if (static_cast<int>(g.models.size()) == 1) {
                    (*lens_out)[0] = g.models[0].len;
                    (*payload_out)[0] = g.payloads[0];
                    return;
                }
                for (int l = 0; l < std::min(streams, static_cast<int>(g.models.size())); ++l) {
                    (*lens_out)[static_cast<std::size_t>(l)] = g.models[static_cast<std::size_t>(l)].len;
                    (*payload_out)[static_cast<std::size_t>(l)] = g.payloads[static_cast<std::size_t>(l)];
                }
            };
            fill_group(comp.root, &buf.lens_root, &buf.payload_root);
            fill_group(comp.linkage, &buf.lens_linkage, &buf.payload_linkage);

            if (!coeff_writer.WriteClusterAt(cid,
                                             comp.scale_root,
                                             comp.scale_linkage,
                                             buf.lens_root,
                                             buf.lens_linkage,
                                             buf.payload_root,
                                             buf.payload_linkage,
                                             &local_err)) {
                fail(local_err);
                return false;
            }
            return true;
        };

        const int nth = std::max(1, omp_get_max_threads());
#pragma omp parallel num_threads(nth) default(none) \
        shared(ok, first_err, err_mu, linkage_list, coeff_writer, coeff_meta, coeff_pre, coeff_scales_ones, coeff_lens_zeros, coeff_payload_empty, cfg, train) \
        firstprivate(d, m, m_codes, nlist, process_cluster)
        {
            ClusterWorkBuf buf;
            std::string local_err;
            io::LinkageListThreadReader linkage_thr;
            if (!linkage_thr.OpenFrom(linkage_list, &local_err)) {
                ok.store(false, std::memory_order_relaxed);
                std::lock_guard<std::mutex> guard(err_mu);
                if (first_err.empty()) {
                    first_err = local_err.empty()
                                    ? "RebuildLinkageCoeffCodecFromLinkageListVirtual: failed to open thread reader."
                                    : local_err;
                }
            }

#pragma omp for schedule(dynamic, 1)
            for (int cid = 0; cid < nlist; ++cid) {
                if (!ok.load(std::memory_order_relaxed)) continue;
                (void)process_cluster(cid, linkage_thr, buf, local_err);
            }
        }

        if (!ok.load(std::memory_order_relaxed)) {
            if (err) *err = first_err;
            return false;
        }
        if (!coeff_writer.Finish(err)) {
            return false;
        }
        return true;
    }
} // namespace stlq
