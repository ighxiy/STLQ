#include "stlq/quantizer/update_codebooks_streaming.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <utility>
#include <vector>

#include <omp.h>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#include "stlq/core/kernel_provider.h"
#include "stlq/core/kernel_provider_cpu.h"
#include "stlq/core/lapack.h"
#include "stlq/core/threading.h"
#include "stlq/io/base_list_store.h"
#include "stlq/io/base_store.h"
#include "stlq/io/dataset_reader_factory.h"
#include "stlq/io/linkage_list_store.h"
#include "stlq/io/bvecs_reader.h"
#include "stlq/io/fbin_reader.h"
#include "stlq/io/fvecs_reader.h"
#include "stlq/io/ivf_lists.h"
#include "stlq/common/logger.h"
#include "stlq/quantizer/spkmeans.h"
#include "stlq/common/timer.h"

namespace stlq
{
    namespace
    {
        constexpr float kEps = 1e-6f;
        std::atomic<std::uint64_t> g_codebook_build_tag{0};

        bool ReaderCanReadU8(const io::DatasetVectorReader* reader) {
            return reader && reader->format == io::VectorFileFormat::kBvecs;
        }

        bool ReaderCanReadF32(const io::DatasetVectorReader* reader) {
            return reader && (reader->format == io::VectorFileFormat::kFvecs ||
                reader->format == io::VectorFileFormat::kFbin);
        }

        inline int Ctz32NonZero(std::uint32_t x) {
            // Caller must ensure x != 0.
#if defined(_MSC_VER)
    unsigned long idx = 0;
    _BitScanForward(&idx, static_cast<unsigned long>(x));
    return static_cast<int>(idx);
#else
            return __builtin_ctz(x);
#endif
        }

        inline void BumpCodebookBuildTag(CodebookPack* pack) {
            if (!pack) return;
            pack->build_tag = g_codebook_build_tag.fetch_add(1, std::memory_order_relaxed) + 1;
        }

        inline std::string FormatDoubleLocal(double value, int precision) {
            std::ostringstream oss;
            oss.setf(std::ios::fixed);
            oss << std::setprecision(std::max(0, precision)) << value;
            return oss.str();
        }

        // Root-layer codes are stored as a byte blob (u8/u16/u32) in linkage_list `code0_one.bin` for init-linkage,
        // and as u8 for train/baseset linkage_list (C_one[0] codes).
        //
        // Use a typed view so the hot loops can read without per-element memcpy or repeated size checks.
        struct RootCodeView
        {
            int size_bytes = 1;
            const std::uint8_t* bytes = nullptr;

            static RootCodeView FromBytes(const std::vector<std::uint8_t>& bytes, int code0_width_bytes) {
                RootCodeView out;
                out.size_bytes = code0_width_bytes;
                out.bytes = bytes.data();
                return out;
            }

            [[nodiscard]] inline int Get(std::size_t idx) const {
                if (size_bytes == 1) {
                    return static_cast<int>(bytes[idx]);
                }
                if (size_bytes == 2) {
                    const std::size_t off = idx * 2u;
                    const auto lo = static_cast<std::uint32_t>(bytes[off]);
                    const std::uint32_t hi = static_cast<std::uint32_t>(bytes[off + 1u]) << 8u;
                    return static_cast<int>(lo | hi);
                }
                const std::size_t off = idx * 4u;
                const auto b0 = static_cast<std::uint32_t>(bytes[off]);
                const std::uint32_t b1 = static_cast<std::uint32_t>(bytes[off + 1u]) << 8u;
                const std::uint32_t b2 = static_cast<std::uint32_t>(bytes[off + 2u]) << 16u;
                const std::uint32_t b3 = static_cast<std::uint32_t>(bytes[off + 3u]) << 24u;
                return static_cast<int>(b0 | b1 | b2 | b3);
            }
        };

        inline bool ValidateLinkageListCodeLayout(const io::LinkageListReader& linkage_list,
                                                  bool allow_wide_code0,
                                                  const char* fn,
                                                  int* code0_width_bytes_out,
                                                  std::string* err) {
            const int small_code_width_bytes = linkage_list.small_code_width_bytes();
            const int code0_width_bytes = linkage_list.code0_width_bytes();
            if (small_code_width_bytes != 1) {
                if (err) {
                    *err = std::string(fn) + ": only u8 small-layer codes are supported (small_code_width_bytes=1).";
                }
                return false;
            }
            if (allow_wide_code0) {
                if (code0_width_bytes != 1 && code0_width_bytes != 2 && code0_width_bytes != 4) {
                    if (err) {
                        *err = std::string(fn) + ": invalid code0_width_bytes (expected 1/2/4).";
                    }
                    return false;
                }
            }
            else if (code0_width_bytes != 1) {
                if (err) {
                    *err = std::string(fn) + ": only uint8 code0_one is supported outside init-linkage legacy stores.";
                }
                return false;
            }
            if (code0_width_bytes_out) {
                *code0_width_bytes_out = code0_width_bytes;
            }
            return true;
        }

        inline void AccumulateReconDepth0FromSmallU8(const CodebookPack& C_root,
                                                     int root_code,
                                                     const std::uint8_t* codes_small_u8, // length m_codes
                                                     const float* coeff_small, // length m_codes
                                                     float a0,
                                                     int m,
                                                     int d,
                                                     float* out) {
            std::fill(out, out + d, 0.0f);

            const float* c0 = C_root.books[0].Col(root_code);
            for (int r = 0; r < d; ++r) {
                out[r] = a0 * c0[r];
            }
            for (int l = 1; l < m; ++l) {
                const int idx = l - 1;
                const int code = static_cast<int>(codes_small_u8[static_cast<std::size_t>(idx)]);
                const float a = coeff_small[static_cast<std::size_t>(idx)];
                const float* center = C_root.books[l].Col(code);
                for (int r = 0; r < d; ++r) {
                    out[r] += a * center[r];
                }
            }
        }

        inline void AccumulateReconDepthPosFromSmallU8(const CodebookPack& C_one,
                                                       int code0_one,
                                                       const std::uint8_t* codes_small_u8, // length m_codes
                                                       const float* coeff_small, // length m_codes
                                                       float a0,
                                                       int m,
                                                       int d,
                                                       float* out) {
            std::fill(out, out + d, 0.0f);

            const float* c0 = C_one.books[0].Col(code0_one);
            for (int r = 0; r < d; ++r) {
                out[r] = a0 * c0[r];
            }
            for (int l = 1; l < m; ++l) {
                const int idx = l - 1;
                const int code = static_cast<int>(codes_small_u8[static_cast<std::size_t>(idx)]);
                const float a = coeff_small[static_cast<std::size_t>(idx)];
                const float* center = C_one.books[l].Col(code);
                for (int r = 0; r < d; ++r) {
                    out[r] += a * center[r];
                }
            }
        }

        struct GtSmall
        {
            int H = 0;
            int d = 0;
            std::vector<int> offsets_small; // size m, offsets_small[l] valid for l>=1
            // Column-major G (H×H) and T (H×d).
            std::vector<float> G;
            std::vector<float> T;
        };

        GtSmall MakeGtSmall(const std::vector<int>& h_vec, int d) {
            const int m = static_cast<int>(h_vec.size());
            GtSmall out;
            out.d = d;
            out.offsets_small.assign(static_cast<std::size_t>(m), 0);
            int H = 0;
            for (int l = 1; l < m; ++l) {
                out.offsets_small[static_cast<std::size_t>(l)] = H;
                H += std::max(0, h_vec[static_cast<std::size_t>(l)]);
            }
            out.H = H;
            out.G.assign(static_cast<std::size_t>(H) * static_cast<std::size_t>(H), 0.0f);
            out.T.assign(static_cast<std::size_t>(H) * static_cast<std::size_t>(d), 0.0f);
            return out;
        }

        inline float& Gcm(std::vector<float>& G, int H, int r, int c) {
            return G[static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * static_cast<std::size_t>(H)];
        }

        inline float& Tcm(std::vector<float>& T, int H, int d, int r, int c) {
            (void)d;
            return T[static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * static_cast<std::size_t>(H)];
        }

        bool SolveGtSmallInPlace(GtSmall* gt, std::vector<float>* W_out, std::string* err) {
            if (!gt || !W_out) {
                if (err) *err = "SolveGtSmallInPlace: null inputs.";
                return false;
            }
            const int H = gt->H;
            const int d = gt->d;
            if (H <= 0 || d <= 0) {
                W_out->clear();
                return true;
            }

            // Regularize diagonal and factorize.
            std::vector<float> Gbak = gt->G;
            for (int i = 0; i < H; ++i) {
                Gcm(gt->G, H, i, i) += kEps;
            }
            int info = lapack::SpotrfU(H, gt->G.data(), H);
            if (info != 0) {
                float bump = kEps;
                for (int attempt = 0; attempt < 10 && info != 0; ++attempt) {
                    gt->G = Gbak;
                    for (int i = 0; i < H; ++i) {
                        Gcm(gt->G, H, i, i) += bump;
                    }
                    info = lapack::SpotrfU(H, gt->G.data(), H);
                    bump *= 10.0f;
                }
                if (info != 0) {
                    if (err) {
                        *err = "SolveGtSmallInPlace: potrf failed (info=" + std::to_string(info) + ").";
                    }
                    return false;
                }
            }

            // Solve G * W = T (W has shape H×d in column-major, stored in T itself).
            info = lapack::SpotrsU(H, d, gt->G.data(), H, gt->T.data(), H);
            if (info != 0) {
                if (err) {
                    *err = "SolveGtSmallInPlace: potrs failed (info=" + std::to_string(info) + ").";
                }
                return false;
            }
            *W_out = std::move(gt->T);
            return true;
        }

        inline int FindListPos(const std::vector<std::pair<std::uint32_t, int>>& gid_pos_sorted,
                               std::uint32_t gid) {
            auto it = std::lower_bound(gid_pos_sorted.begin(), gid_pos_sorted.end(),
                                       std::pair<std::uint32_t, int>(gid, 0),
                                       [](const auto& a, const auto& b) { return a.first < b.first; });
            if (it == gid_pos_sorted.end() || it->first != gid) {
                return -1;
            }
            return it->second;
        }
    } // namespace

    bool UpdateCRootFromTrainBasicStreaming(const Config& cfg,
                                            const std::string& train_basic_dir,
                                            const ColMajorMatrix<float>& R,
                                            const io::DatasetVectorReader* train_reader,
                                            StreamKernelProvider* kernels,
                                            CodebookPack* C_root_inout,
                                            float* out_mse_after_update,
                                            std::string* err) {
        if (!kernels) {
            if (err) *err = "UpdateCRootFromTrainBasicStreaming: kernels is null.";
            return false;
        }
        if (!C_root_inout) {
            if (err) *err = "UpdateCRootFromTrainBasicStreaming: C_root output is null.";
            return false;
        }
        if (train_basic_dir.empty()) {
            if (err) *err = "UpdateCRootFromTrainBasicStreaming: empty train_basic_dir.";
            return false;
        }
        if (C_root_inout->books.empty()) {
            if (err) *err = "UpdateCRootFromTrainBasicStreaming: empty C_root.";
            return false;
        }

        io::BaseBasicReader basic;
        if (!basic.Open(train_basic_dir, err)) {
            return false;
        }
        const int d = basic.meta().d;
        const int m = cfg.model.m;
        if (d <= 0 || m <= 1 || static_cast<int>(cfg.model.h_vec.size()) != m) {
            if (err) *err = "UpdateCRootFromTrainBasicStreaming: invalid d/m/h_vec.";
            return false;
        }
        if (R.rows != d || R.cols != d) {
            if (err) *err = "UpdateCRootFromTrainBasicStreaming: rotation shape mismatch.";
            return false;
        }
        const int h0 = cfg.model.h_vec[0];
        if (h0 <= 0 || C_root_inout->books[0].cols != h0 || C_root_inout->books[0].rows != d) {
            if (err) *err = "UpdateCRootFromTrainBasicStreaming: C_root[0] shape mismatch.";
            return false;
        }

        const std::uint64_t n = basic.meta().write_basic_to_bucket ? basic.meta().shard_size : 0;
        (void)n;

        std::uint64_t ntrain = train_reader ? train_reader->n : 0;
        if (cfg.dataset.ntrain_set && cfg.dataset.ntrain > 0) {
            ntrain = std::min<std::uint64_t>(ntrain, static_cast<std::uint64_t>(cfg.dataset.ntrain));
        }
        if (ntrain == 0) {
            if (err) *err = "UpdateCRootFromTrainBasicStreaming: invalid ntrain.";
            return false;
        }

        GtSmall gt = MakeGtSmall(cfg.model.h_vec, d);
        const int Hs = gt.H;

        // Pass 1: accumulate G/T for small layers using x_res = x_rot - a0*c0.
        const std::uint32_t block = static_cast<std::uint32_t>(std::max(1, cfg.large.train_block));
        std::vector<std::uint32_t> cluster_id;
        ColMajorMatrix<Code> B_small;
        ColMajorMatrix<float> a_all;
        ColMajorMatrix<std::uint8_t> x_u8;
        ColMajorMatrix<float> x_f32;
        ColMajorMatrix<float> Xrot;
        std::vector<float> x_res(static_cast<std::size_t>(d), 0.0f);
        auto read_train_block_rotated = [&](std::uint64_t start,
                                            std::uint32_t count,
                                            const char* caller) -> bool
        {
            if (!train_reader) {
                if (err) *err = std::string(caller) + ": no train reader provided.";
                return false;
            }
            if (ReaderCanReadU8(train_reader)) {
                if (!io::ReadDatasetVectorBlockU8(*train_reader, start, count, &x_u8, err)) {
                    return false;
                }
                kernels->ConvertU8ToF32AndRotate(x_u8, R, &Xrot);
                return true;
            }
            if (ReaderCanReadF32(train_reader)) {
                if (!io::ReadDatasetVectorBlockF32(*train_reader, start, count, &x_f32, err)) {
                    return false;
                }
                Xrot.rows = d;
                Xrot.cols = x_f32.cols;
                Xrot.data.resize(static_cast<std::size_t>(d) * static_cast<std::size_t>(x_f32.cols));
                kernels->Gemm(false, false, 1.0f, R, x_f32, 0.0f, &Xrot);
                return true;
            }
            if (err) *err = std::string(caller) + ": unsupported train reader format.";
            return false;
        };

        for (std::uint64_t start = 0; start < ntrain; start += block) {
            const std::uint32_t count =
                static_cast<std::uint32_t>(std::min<std::uint64_t>(block, ntrain - start));
            if (!basic.ReadClusterIdBlock(start, count, &cluster_id, err)) {
                return false;
            }
            if (!basic.ReadCodesBlock(start, count, &B_small, err)) {
                return false;
            }
            if (!basic.ReadCoeffsBlock(start, count, &a_all, err)) {
                return false;
            }
            if (!read_train_block_rotated(start, count, "UpdateCRootFromTrainBasicStreaming")) {
                return false;
            }

            for (std::uint32_t j = 0; j < count; ++j) {
                const int cid = static_cast<int>(cluster_id[static_cast<std::size_t>(j)]);
                const float a0 = a_all(0, static_cast<int>(j));
                const float* c0 = C_root_inout->books[0].Col(cid);
                const float* x = Xrot.Col(static_cast<int>(j));
                for (int r = 0; r < d; ++r) {
                    x_res[static_cast<std::size_t>(r)] = x[r] - a0 * c0[r];
                }

                int pidx[64];
                float alpha[64];
                int used = 0;
                for (int l = 1; l < m; ++l) {
                    const int code = static_cast<int>(B_small(l - 1, static_cast<int>(j)));
                    const float al = a_all(l, static_cast<int>(j));
                    const int p = gt.offsets_small[static_cast<std::size_t>(l)] + code;
                    pidx[used] = p;
                    alpha[used] = al;
                    ++used;
                }

                for (int t = 0; t < used; ++t) {
                    const int p = pidx[t];
                    const float at = alpha[t];
                    for (int r = 0; r < d; ++r) {
                        Tcm(gt.T, Hs, d, p, r) += at * x_res[static_cast<std::size_t>(r)];
                    }
                }
                for (int a = 0; a < used; ++a) {
                    const int pa = pidx[a];
                    const float aa = alpha[a];
                    for (int b = a; b < used; ++b) {
                        const int pb = pidx[b];
                        const float v = aa * alpha[b];
                        const int r0 = std::min(pa, pb);
                        const int c0i = std::max(pa, pb);
                        Gcm(gt.G, Hs, r0, c0i) += v;
                    }
                }
            }
        }

        std::vector<float> W_small;
        std::string local_err;
        if (!SolveGtSmallInPlace(&gt, &W_small, &local_err)) {
            if (err) *err = "UpdateCRootFromTrainBasicStreaming: small-layer solve failed: " + local_err;
            return false;
        }

        // Write back small layers.
        for (int l = 1; l < m; ++l) {
            const int hl = cfg.model.h_vec[static_cast<std::size_t>(l)];
            ColMajorMatrix<float> book(d, hl);
            for (int code = 0; code < hl; ++code) {
                const int p = gt.offsets_small[static_cast<std::size_t>(l)] + code;
                float* dst = book.Col(code);
                for (int r = 0; r < d; ++r) {
                    dst[r] = W_small[static_cast<std::size_t>(p) + static_cast<std::size_t>(r) * static_cast<
                        std::size_t>(Hs)];
                }
            }
            C_root_inout->books[static_cast<std::size_t>(l)] = std::move(book);
        }

        // Pass 2: closed-form update for root layer using updated small layers.
        std::vector<double> num0(static_cast<std::size_t>(d) * static_cast<std::size_t>(h0), 0.0);
        std::vector<double> den0(static_cast<std::size_t>(h0), 0.0);
        std::vector<float> recon_small(static_cast<std::size_t>(d), 0.0f);

        for (std::uint64_t start = 0; start < ntrain; start += block) {
            const std::uint32_t count =
                static_cast<std::uint32_t>(std::min<std::uint64_t>(block, ntrain - start));
            if (!basic.ReadClusterIdBlock(start, count, &cluster_id, err)) {
                return false;
            }
            if (!basic.ReadCodesBlock(start, count, &B_small, err)) {
                return false;
            }
            if (!basic.ReadCoeffsBlock(start, count, &a_all, err)) {
                return false;
            }
            if (!read_train_block_rotated(start, count, "UpdateCRootFromTrainBasicStreaming")) {
                return false;
            }

            for (std::uint32_t j = 0; j < count; ++j) {
                const int cid = static_cast<int>(cluster_id[static_cast<std::size_t>(j)]);
                const float a0 = a_all(0, static_cast<int>(j));
                const float* x = Xrot.Col(static_cast<int>(j));

                std::fill(recon_small.begin(), recon_small.end(), 0.0f);
                for (int l = 1; l < m; ++l) {
                    const int code = static_cast<int>(B_small(l - 1, static_cast<int>(j)));
                    const float al = a_all(l, static_cast<int>(j));
                    const float* center = C_root_inout->books[static_cast<std::size_t>(l)].Col(code);
                    for (int r = 0; r < d; ++r) {
                        recon_small[static_cast<std::size_t>(r)] += al * center[r];
                    }
                }

                const auto w = static_cast<double>(a0);
                den0[static_cast<std::size_t>(cid)] += w * w;
                for (int r = 0; r < d; ++r) {
                    const double v = w * static_cast<double>(x[r] - recon_small[static_cast<std::size_t>(r)]);
                    num0[static_cast<std::size_t>(r) + static_cast<std::size_t>(cid) * static_cast<std::size_t>(d)] +=
                        v;
                }
            }
        }

        ColMajorMatrix<float> book0(d, h0);
        for (int k = 0; k < h0; ++k) {
            const double den = den0[static_cast<std::size_t>(k)];
            float* dst = book0.Col(k);
            if (den <= 0.0) {
                // This root cluster received no assignments. Do NOT overwrite it with zeros;
                // that can permanently kill the cluster downstream (large-root beam search skips alpha==0).
                const float* prev = C_root_inout->books[0].Col(k);
                std::memcpy(dst, prev, static_cast<std::size_t>(d) * sizeof(float));
                continue;
            }
            const double inv = 1.0 / den;
            for (int r = 0; r < d; ++r) {
                dst[r] = static_cast<float>(num0[static_cast<std::size_t>(r) +
                    static_cast<std::size_t>(k) * static_cast<std::size_t>(d)] * inv);
            }
        }
        C_root_inout->books[0] = std::move(book0);

        // Optional: compute training MSE after updating C_root using stored (cluster_id, codes_small, a).
        // This matches the baseline semantic "MSE after updating C_root" without re-encoding.
        if (out_mse_after_update) {
            double sse_sum = 0.0;
            std::uint64_t n_seen = 0;

            for (std::uint64_t start = 0; start < ntrain; start += block) {
                const std::uint32_t count =
                    static_cast<std::uint32_t>(std::min<std::uint64_t>(block, ntrain - start));
                if (!basic.ReadClusterIdBlock(start, count, &cluster_id, err)) {
                    return false;
                }
                if (!basic.ReadCodesBlock(start, count, &B_small, err)) {
                    return false;
                }
                if (!basic.ReadCoeffsBlock(start, count, &a_all, err)) {
                    return false;
                }
                if (!read_train_block_rotated(start, count, "UpdateCRootFromTrainBasicStreaming")) {
                    return false;
                }

                double sse_block = 0.0;
#pragma omp parallel default(none) shared(cluster_id, a_all, Xrot, C_root_inout, B_small) firstprivate(count, d, m) reduction(+:sse_block)
                {
                    std::vector<float> recon(static_cast<std::size_t>(d), 0.0f);
#pragma omp for schedule(static)
                    for (std::int64_t j = 0; j < count; ++j) {
                        const int cid = static_cast<int>(cluster_id[static_cast<std::size_t>(j)]);
                        const float a0 = a_all(0, static_cast<int>(j));
                        const float* x = Xrot.Col(static_cast<int>(j));

                        const float* c0 = C_root_inout->books[0].Col(cid);
                        for (int r = 0; r < d; ++r) {
                            recon[static_cast<std::size_t>(r)] = a0 * c0[r];
                        }
                        for (int l = 1; l < m; ++l) {
                            const int code = static_cast<int>(B_small(l - 1, static_cast<int>(j)));
                            const float al = a_all(l, static_cast<int>(j));
                            const float* center = C_root_inout->books[static_cast<std::size_t>(l)].Col(code);
                            for (int r = 0; r < d; ++r) {
                                recon[static_cast<std::size_t>(r)] += al * center[r];
                            }
                        }

                        double e = 0.0;
                        for (int r = 0; r < d; ++r) {
                            const double diff =
                                static_cast<double>(x[r]) - static_cast<double>(recon[static_cast<std::size_t>(r)]);
                            e += diff * diff;
                        }
                        sse_block += e;
                    }
                }
                sse_sum += sse_block;
                n_seen += static_cast<std::uint64_t>(count);
            }
            *out_mse_after_update = (n_seen > 0) ? static_cast<float>(sse_sum / static_cast<double>(n_seen)) : 0.0f;
        }
        return true;
    }

    // Strictly-equivalent streaming update for C_root.
    //
    // Mathematical sketch (fixed codes B and coefficients a):
    //   Min over {c0[cluster], v[p]}  Σ_i || x_i - a0_i*c0[gid_i] - Σ_p a_{i,p} v[p] ||^2
    // Let g_c[p] = Σ_{i in c} a0_i * a_{i,p},  den_c = Σ_{i in c} a0_i^2,  t0_c = Σ_{i in c} a0_i x_i,
    // and (Gss,Tss) be the normal-equation stats for small layers:
    //   Gss[p,q] = Σ_i a_{i,p} a_{i,q},  Tss[p] = Σ_i a_{i,p} x_i.
    // Eliminating c0 per cluster yields the Schur system:
    //   (Gss + λI - Σ_c g_c g_c^T /(den_c+λ)) * V = (Tss - Σ_c g_c t0_c^T /(den_c+λ)),
    // where λ is the same diagonal "bump" as the full in-memory solve (0, 1e-6, 1e-5, ... if needed).
    // After solving V, recover:
    //   c0_c = (t0_c - Σ_p g_c[p] V_p) / (den_c+λ).
    //
    // This function implements the exact Schur complement solve in streaming form, avoiding construction of the
    // (h0+Hs)×(h0+Hs) Gram matrix but matching the full LS optimum (up to float rounding and identical bump policy).
    bool UpdateCRootFromTrainBasicStreamingExactLS(const Config& cfg,
                                                   const std::string& train_basic_dir,
                                                   const ColMajorMatrix<float>& R,
                                                   const io::DatasetVectorReader* train_reader,
                                                   StreamKernelProvider* kernels,
                                                   CodebookPack* C_root_inout,
                                                   float* out_mse_after_update,
                                                   double* out_mse_wall_sec,
                                                   std::string* err) {
        if (!kernels) {
            if (err) *err = "UpdateCRootFromTrainBasicStreamingExactLS: kernels is null.";
            return false;
        }
        if (!C_root_inout) {
            if (err) *err = "UpdateCRootFromTrainBasicStreamingExactLS: C_root output is null.";
            return false;
        }
        if (out_mse_wall_sec) {
            *out_mse_wall_sec = 0.0;
        }
        if (train_basic_dir.empty()) {
            if (err) *err = "UpdateCRootFromTrainBasicStreamingExactLS: empty train_basic_dir.";
            return false;
        }
        if (C_root_inout->books.empty()) {
            if (err) *err = "UpdateCRootFromTrainBasicStreamingExactLS: empty C_root.";
            return false;
        }

        io::BaseBasicReader basic;
        if (!basic.Open(train_basic_dir, err)) {
            return false;
        }
        const int d = basic.meta().d;
        const int m = cfg.model.m;
        const int s = std::max(0, m - 1);
        if (d <= 0 || m <= 1 || static_cast<int>(cfg.model.h_vec.size()) != m) {
            if (err) *err = "UpdateCRootFromTrainBasicStreamingExactLS: invalid d/m/h_vec.";
            return false;
        }
        if (R.rows != d || R.cols != d) {
            if (err) *err = "UpdateCRootFromTrainBasicStreamingExactLS: rotation shape mismatch.";
            return false;
        }

        const int h0 = cfg.model.h_vec[0];
        if (h0 <= 0 || static_cast<int>(C_root_inout->books.size()) != m ||
            C_root_inout->books[0].cols != h0 || C_root_inout->books[0].rows != d) {
            if (err) *err = "UpdateCRootFromTrainBasicStreamingExactLS: C_root shape mismatch.";
            return false;
        }

        const int nth_target = (cfg.runtime.omp_threads > 0) ? cfg.runtime.omp_threads : omp_get_max_threads();
        ScopedOmpThreads scoped_omp(nth_target);

        // This optimized implementation assumes all small layers have <=256 codewords so we can use a compact 256-bit mask
        // per (cluster, layer) to enumerate nonzeros. This matches the intended large-scale configuration:
        //   h0 up to 65536, h_l=256 for l>=1, d=128, m<=16.
        for (int l = 1; l < m; ++l) {
            if (cfg.model.h_vec[static_cast<std::size_t>(l)] > 256) {
                if (err)
                    *err =
                        "UpdateCRootFromTrainBasicStreamingExactLS: requires h_l<=256 for l>=1 in optimized path.";
                return false;
            }
        }

        std::uint64_t ntrain = train_reader ? train_reader->n : 0;
        if (cfg.dataset.ntrain_set && cfg.dataset.ntrain > 0) {
            ntrain = std::min<std::uint64_t>(ntrain, static_cast<std::uint64_t>(cfg.dataset.ntrain));
        }
        if (ntrain == 0) {
            if (err) *err = "UpdateCRootFromTrainBasicStreamingExactLS: ntrain is zero.";
            return false;
        }

        // Flattening meta for small layers.
        GtSmall gt = MakeGtSmall(cfg.model.h_vec, d);
        const int Hs = gt.H;
        if (Hs <= 0) {
            // Degenerate: no small layers; root reduces to per-cluster closed form.
            // We still need den0/t0, then c0 = t0/(den0+λ) (λ bump handled implicitly by kEps).
            std::vector<double> den0_d(static_cast<std::size_t>(h0), 0.0);
            std::vector<double> t0_d(static_cast<std::size_t>(h0) * static_cast<std::size_t>(d), 0.0);

            const std::uint32_t block = static_cast<std::uint32_t>(std::max(1, cfg.large.train_block));
            std::vector<std::uint32_t> cluster_id;
            ColMajorMatrix<float> a_all;
            ColMajorMatrix<std::uint8_t> x_u8;
            ColMajorMatrix<float> x_f32;
            ColMajorMatrix<float> Xrot;

            for (std::uint64_t start = 0; start < ntrain; start += block) {
                const std::uint32_t count =
                    static_cast<std::uint32_t>(std::min<std::uint64_t>(block, ntrain - start));
                if (!basic.ReadClusterIdBlock(start, count, &cluster_id, err)) {
                    return false;
                }
                if (!basic.ReadCoeffsBlock(start, count, &a_all, err)) {
                    return false;
                }
                if (!train_reader) {
                    if (err) *err = "UpdateCRootFromTrainBasicStreamingExactLS: no train reader provided.";
                    return false;
                }
                if (ReaderCanReadU8(train_reader)) {
                    if (!io::ReadDatasetVectorBlockU8(*train_reader, start, count, &x_u8, err)) {
                        return false;
                    }
                    kernels->ConvertU8ToF32AndRotate(x_u8, R, &Xrot);
                }
                else if (ReaderCanReadF32(train_reader)) {
                    if (!io::ReadDatasetVectorBlockF32(*train_reader, start, count, &x_f32, err)) {
                        return false;
                    }
                    Xrot.rows = d;
                    Xrot.cols = x_f32.cols;
                    Xrot.data.resize(static_cast<std::size_t>(d) * static_cast<std::size_t>(x_f32.cols));
                    kernels->Gemm(false, false, 1.0f, R, x_f32, 0.0f, &Xrot);
                }
                else {
                    if (err) *err = "UpdateCRootFromTrainBasicStreamingExactLS: unsupported train reader format.";
                    return false;
                }

                for (std::uint32_t j = 0; j < count; ++j) {
                    const int cid = static_cast<int>(cluster_id[static_cast<std::size_t>(j)]);
                    const float a0 = a_all(0, static_cast<int>(j));
                    const float* x = Xrot.Col(static_cast<int>(j));
                    den0_d[static_cast<std::size_t>(cid)] += static_cast<double>(a0) * static_cast<double>(a0);
                    for (int r = 0; r < d; ++r) {
                        t0_d[static_cast<std::size_t>(cid) * static_cast<std::size_t>(d) + static_cast<std::size_t>(r)]
                            +=
                            static_cast<double>(a0) * static_cast<double>(x[r]);
                    }
                }
            }

            ColMajorMatrix<float> book0(d, h0);
            for (int cid = 0; cid < h0; ++cid) {
                const double denom = den0_d[static_cast<std::size_t>(cid)];
                float* c0 = book0.Col(cid);
                if (denom <= 0.0) {
                    // Unassigned cluster: preserve previous root vector instead of writing zeros.
                    const float* prev = C_root_inout->books[0].Col(cid);
                    std::memcpy(c0, prev, static_cast<std::size_t>(d) * sizeof(float));
                    continue;
                }
                const double inv = 1.0 / denom;
                for (int r = 0; r < d; ++r) {
                    c0[r] = static_cast<float>(t0_d[static_cast<std::size_t>(cid) * static_cast<std::size_t>(d) +
                        static_cast<std::size_t>(r)] * inv);
                }
            }
            C_root_inout->books[0] = std::move(book0);
            if (out_mse_after_update) {
                *out_mse_after_update = 0.0f; // caller likely not using this degenerate path
            }
            BumpCodebookBuildTag(C_root_inout);
            return true;
        }

        // ---- Phase A: streaming accumulation (double -> float system) ----
        const bool profile = cfg.large.profile_timing;
        const double prof_total_t0 = profile ? omp_get_wtime() : 0.0;
        double prof_read_basic_sec = 0.0;
        double prof_read_x_sec = 0.0;
        double prof_convert_rotate_sec = 0.0;
        double prof_partition_sec = 0.0;
        double prof_accum_sec = 0.0;
        double prof_reduce_stats_sec = 0.0;
        double prof_cast_stats_sec = 0.0;
        double prof_schur_accum_sec = 0.0;
        double prof_schur_reduce_sec = 0.0;
        double prof_schur_build_sec = 0.0;
        double prof_schur_solve_sec = 0.0;
        double prof_write_small_sec = 0.0;
        double prof_recover_root_sec = 0.0;

        const std::uint32_t block = static_cast<std::uint32_t>(std::max(1, cfg.large.train_block));

        if (profile) {
            LogInfo("Update C_root (profile): provider=" + std::string(kernels->IsGpu() ? "cuda" : "cpu") +
                " block=" + std::to_string(block) + " ntrain=" + std::to_string(ntrain) +
                " h0=" + std::to_string(h0) +
                " Hs=" + std::to_string(Hs) + " d=" + std::to_string(d) + " m=" + std::to_string(m) +
                " omp_max=" + std::to_string(omp_get_max_threads()));
        }
        // Gss/Tss are small-layer normal-equation stats.
        // Gss is symmetric; keep per-thread packed upper triangle to avoid contention without huge memory blowups.
        const int nth_phase_a = std::max(1, omp_get_max_threads());
        // Partition cids into contiguous ranges across the OpenMP team to make per-cluster stats contention-free.
        const int cids_per_part = std::max(1, (h0 + nth_phase_a - 1) / nth_phase_a);
        const std::size_t packed_phase_a = static_cast<std::size_t>(Hs) * static_cast<std::size_t>(Hs + 1) / 2u;
        const std::size_t td_size_phase_a = static_cast<std::size_t>(Hs) * static_cast<std::size_t>(d);
        // NOTE: allocate without value-initialization, then first-touch/zero in parallel.
        // This avoids NUMA/remote-memory pathologies when running with many OpenMP threads on multi-socket hosts.
        const std::size_t Gss_packed_all_size = packed_phase_a * static_cast<std::size_t>(nth_phase_a);
        auto Gss_packed_all_u = std::unique_ptr<double[]>(new double[Gss_packed_all_size]);
        double* Gss_packed_all = Gss_packed_all_u.get();
        // NOTE: store Tss in transposed layout per thread to avoid strided writes:
        //   Tss_t[p, r] (p=0..Hs-1, r=0..d-1) is stored as contiguous d for each p (row-major blocks).
        // This makes the hot-loop update "add scaled x[r]" contiguous in memory.
        const std::size_t Tss_all_t_size = td_size_phase_a * static_cast<std::size_t>(nth_phase_a);
        auto Tss_all_t_u = std::unique_ptr<double[]>(new double[Tss_all_t_size]);
        double* Tss_all_t = Tss_all_t_u.get();

        // Root stats (partitioned by cid-range / OpenMP thread).
        // Layout: [tid * cids_per_part + (cid - tid*cids_per_part)] for den0,
        //         [same * d + r] for t0.
        const std::size_t den0_part_size = static_cast<std::size_t>(cids_per_part) * static_cast<std::size_t>(
            nth_phase_a);
        const std::size_t t0_part_size = den0_part_size * static_cast<std::size_t>(d);
        auto den0_part_u = std::unique_ptr<double[]>(new double[den0_part_size]);
        auto t0_part_u = std::unique_ptr<double[]>(new double[t0_part_size]);
        double* den0_part = den0_part_u.get();
        double* t0_part = t0_part_u.get();

        // Cross stats g_c[p] = Σ_{i in c} a0_i * a_{i,p}.
        // Optimized representation:
        //  - g_hist: dense float [h0, Hs], ~604MB at h0=65536, Hs=2304 (m=10).
        //  - g_mask: 256-bit mask per (cluster, small-layer) to enumerate nonzeros quickly.
        const std::size_t g_hist_size = static_cast<std::size_t>(h0) * static_cast<std::size_t>(Hs);
        auto g_hist_u = std::unique_ptr<float[]>(new float[g_hist_size]);
        float* g_hist = g_hist_u.get();
        constexpr int kMaskWords = 8; // 256 bits / 32
        const std::size_t g_mask_size = static_cast<std::size_t>(h0) * static_cast<std::size_t>(s) * kMaskWords;
        auto g_mask_u = std::unique_ptr<std::uint32_t[]>(new std::uint32_t[g_mask_size]);
        std::uint32_t* g_mask = g_mask_u.get();

        auto Td = [](std::vector<double>& T, int H, int dloc, int r, int c) -> double& {
            (void)dloc;
            return T[static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * static_cast<std::size_t>(H)];
        };
        auto PackUpper = [](int i, int j) -> std::size_t
        {
            // i <= j
            return static_cast<std::size_t>(j) * static_cast<std::size_t>(j + 1) / 2u + static_cast<std::size_t>(i);
        };

        std::vector<std::uint32_t> cluster_id;
        ColMajorMatrix<Code> B_small;
        ColMajorMatrix<float> a_all;
        ColMajorMatrix<std::uint8_t> x_u8;
        ColMajorMatrix<float> x_f32;
        ColMajorMatrix<float> Xrot;
        auto read_train_block_rotated_exact = [&](std::uint64_t start,
                                                  std::uint32_t count) -> bool
        {
            if (!train_reader) {
                if (err) *err = "UpdateCRootFromTrainBasicStreamingExactLS: no train reader provided.";
                return false;
            }
            if (ReaderCanReadU8(train_reader)) {
                if (profile) {
                    Timer t;
                    if (!io::ReadDatasetVectorBlockU8(*train_reader, start, count, &x_u8, err)) {
                        return false;
                    }
                    prof_read_x_sec += t.ElapsedSeconds();
                    t = Timer();
                    kernels->ConvertU8ToF32AndRotate(x_u8, R, &Xrot);
                    prof_convert_rotate_sec += t.ElapsedSeconds();
                }
                else {
                    if (!io::ReadDatasetVectorBlockU8(*train_reader, start, count, &x_u8, err)) {
                        return false;
                    }
                    kernels->ConvertU8ToF32AndRotate(x_u8, R, &Xrot);
                }
                return true;
            }
            if (ReaderCanReadF32(train_reader)) {
                if (profile) {
                    Timer t;
                    if (!io::ReadDatasetVectorBlockF32(*train_reader, start, count, &x_f32, err)) {
                        return false;
                    }
                    prof_read_x_sec += t.ElapsedSeconds();
                    Xrot.rows = d;
                    Xrot.cols = x_f32.cols;
                    Xrot.data.resize(static_cast<std::size_t>(d) * static_cast<std::size_t>(x_f32.cols));
                    t = Timer();
                    kernels->Gemm(false, false, 1.0f, R, x_f32, 0.0f, &Xrot);
                    prof_convert_rotate_sec += t.ElapsedSeconds();
                }
                else {
                    if (!io::ReadDatasetVectorBlockF32(*train_reader, start, count, &x_f32, err)) {
                        return false;
                    }
                    Xrot.rows = d;
                    Xrot.cols = x_f32.cols;
                    Xrot.data.resize(static_cast<std::size_t>(d) * static_cast<std::size_t>(x_f32.cols));
                    kernels->Gemm(false, false, 1.0f, R, x_f32, 0.0f, &Xrot);
                }
                return true;
            }
            if (err) *err = "UpdateCRootFromTrainBasicStreamingExactLS: unsupported train reader format.";
            return false;
        };


        std::vector<std::uint32_t> part_counts(static_cast<std::size_t>(nth_phase_a), 0u);
        std::vector<std::uint32_t> part_offsets(static_cast<std::size_t>(nth_phase_a) + 1u, 0u);
        std::vector<std::uint32_t> part_cursor(static_cast<std::size_t>(nth_phase_a), 0u);
        std::vector<std::uint32_t> part_indices;

        // First-touch/zero large accumulators in parallel to keep them local on multi-socket hosts.
#pragma omp parallel default(none) shared(Gss_packed_all, Tss_all_t, den0_part, t0_part, g_hist, g_mask) firstprivate(packed_phase_a, td_size_phase_a, cids_per_part, d, h0, Hs, s, kMaskWords)
        {
            const int tid = omp_get_thread_num();
            // Per-thread slices.
            std::fill(Gss_packed_all + static_cast<std::size_t>(tid) * packed_phase_a,
                      Gss_packed_all + static_cast<std::size_t>(tid + 1) * packed_phase_a, 0.0);
            std::fill(Tss_all_t + static_cast<std::size_t>(tid) * td_size_phase_a,
                      Tss_all_t + static_cast<std::size_t>(tid + 1) * td_size_phase_a, 0.0);
            std::fill(den0_part + static_cast<std::size_t>(tid) * static_cast<std::size_t>(cids_per_part),
                      den0_part + static_cast<std::size_t>(tid + 1) * static_cast<std::size_t>(cids_per_part), 0.0);
            std::fill(
                t0_part + static_cast<std::size_t>(tid) * static_cast<std::size_t>(cids_per_part) * static_cast<
                    std::size_t>(d),
                t0_part + static_cast<std::size_t>(tid + 1) * static_cast<std::size_t>(cids_per_part) * static_cast<
                    std::size_t>(d), 0.0);

            const int cid_begin = tid * cids_per_part;
            const int cid_end = std::min(h0, cid_begin + cids_per_part);
            if (cid_begin < cid_end) {
                const auto cid_count = static_cast<std::size_t>(cid_end - cid_begin);
                std::fill(g_hist + static_cast<std::size_t>(cid_begin) * static_cast<std::size_t>(Hs),
                          g_hist + (static_cast<std::size_t>(cid_begin) + cid_count) * static_cast<std::size_t>(Hs),
                          0.0f);
                std::fill(g_mask + static_cast<std::size_t>(cid_begin) * static_cast<std::size_t>(s) * kMaskWords,
                          g_mask + (static_cast<std::size_t>(cid_begin) + cid_count) * static_cast<std::size_t>(s) *
                          kMaskWords, 0u);
            }
        }


        for (std::uint64_t start = 0; start < ntrain; start += block) {
            const std::uint32_t count =
                static_cast<std::uint32_t>(std::min<std::uint64_t>(block, ntrain - start));
            if (profile) {
                const Timer t;
                if (!basic.ReadClusterIdBlock(start, count, &cluster_id, err)) {
                    return false;
                }
                prof_read_basic_sec += t.ElapsedSeconds();
            }
            else if (!basic.ReadClusterIdBlock(start, count, &cluster_id, err)) {
                return false;
            }
            if (profile) {
                const Timer t;
                if (!basic.ReadCodesBlock(start, count, &B_small, err)) {
                    return false;
                }
                prof_read_basic_sec += t.ElapsedSeconds();
            }
            else if (!basic.ReadCodesBlock(start, count, &B_small, err)) {
                return false;
            }
            if (profile) {
                const Timer t;
                if (!basic.ReadCoeffsBlock(start, count, &a_all, err)) {
                    return false;
                }
                prof_read_basic_sec += t.ElapsedSeconds();
            }
            else if (!basic.ReadCoeffsBlock(start, count, &a_all, err)) {
                return false;
            }
            if (!read_train_block_rotated_exact(start, count)) {
                return false;
            }

            // Build per-thread index ranges based on cid partition (stable within this block).
            const double t_partition = profile ? omp_get_wtime() : 0.0;
            std::fill(part_counts.begin(), part_counts.end(), 0u);
            for (std::uint32_t j = 0; j < count; ++j) {
                const int cid = static_cast<int>(cluster_id[static_cast<std::size_t>(j)]);
                if (cid < 0 || cid >= h0) {
                    if (err) *err = "UpdateCRootFromTrainBasicStreamingExactLS: cluster_id out of range.";
                    return false;
                }
                const int part = std::min(nth_phase_a - 1, cid / cids_per_part);
                ++part_counts[static_cast<std::size_t>(part)];
            }
            part_offsets[0] = 0u;
            for (int t = 0; t < nth_phase_a; ++t) {
                part_offsets[static_cast<std::size_t>(t) + 1u] =
                    part_offsets[static_cast<std::size_t>(t)] + part_counts[static_cast<std::size_t>(t)];
            }
            part_cursor = part_offsets;
            part_indices.assign(static_cast<std::size_t>(count), 0u);
            for (std::uint32_t j = 0; j < count; ++j) {
                const int cid = static_cast<int>(cluster_id[static_cast<std::size_t>(j)]);
                const int part = std::min(nth_phase_a - 1, cid / cids_per_part);
                const auto at = static_cast<std::size_t>(part_cursor[static_cast<std::size_t>(part)]++);
                part_indices[at] = j;
            }
            if (profile) {
                prof_partition_sec += (omp_get_wtime() - t_partition);
            }

            std::atomic<bool> oob_code{false};
            const double t_accum = profile ? omp_get_wtime() : 0.0;
#pragma omp parallel default(none) shared(Gss_packed_all, Tss_all_t, den0_part, t0_part, part_offsets, part_indices, cluster_id, a_all, Xrot, B_small, cfg, gt, g_hist, g_mask, oob_code, err) firstprivate(packed_phase_a, td_size_phase_a, cids_per_part, d, m, Hs, s, kMaskWords, PackUpper)
            {
                const int tid = omp_get_thread_num();
                double* G_packed =
                    Gss_packed_all + static_cast<std::size_t>(tid) * packed_phase_a;
                double* T_loc_t =
                    Tss_all_t + static_cast<std::size_t>(tid) * td_size_phase_a;

                const std::uint32_t beg = part_offsets[static_cast<std::size_t>(tid)];
                const std::uint32_t end = part_offsets[static_cast<std::size_t>(tid) + 1u];

                for (std::uint32_t kk = beg; kk < end; ++kk) {
                    const std::uint32_t j = part_indices[static_cast<std::size_t>(kk)];
                    const int cid = static_cast<int>(cluster_id[static_cast<std::size_t>(j)]);
                    const float a0 = a_all(0, static_cast<int>(j));
                    const float* x = Xrot.Col(static_cast<int>(j));

                    const std::size_t cbase =
                        static_cast<std::size_t>(tid) * static_cast<std::size_t>(cids_per_part) +
                        static_cast<std::size_t>(cid - tid * cids_per_part);
                    den0_part[cbase] += static_cast<double>(a0) * static_cast<double>(a0);
                    const std::size_t t0_off = cbase * static_cast<std::size_t>(d);

                    for (int r = 0; r < d; ++r) {
                        t0_part[t0_off + static_cast<std::size_t>(r)] +=
                            static_cast<double>(a0) * static_cast<double>(x[r]);
                    }

                    int pidx[32];
                    float alpha[32];
                    int used = 0;
                    const std::size_t base = static_cast<std::size_t>(cid) * static_cast<std::size_t>(Hs);
                    for (int l = 1; l < m; ++l) {
                        if (oob_code.load(std::memory_order_relaxed)) break;
                        const int hl = cfg.model.h_vec[static_cast<std::size_t>(l)];
                        const int code = static_cast<int>(B_small(l - 1, static_cast<int>(j)));
                        if (code < 0 || code >= hl) {
                            if (!oob_code.exchange(true, std::memory_order_relaxed)) {
#pragma omp critical
                                {
                                    if (err) *err = "UpdateCRootFromTrainBasicStreamingExactLS: code out of range.";
                                }
                            }
                            break;
                        }
                        const float al = a_all(l, static_cast<int>(j));
                        const int p = gt.offsets_small[static_cast<std::size_t>(l)] + code;
                        pidx[used] = p;
                        alpha[used] = al;
                        ++used;

                        g_hist[base + static_cast<std::size_t>(p)] += a0 * al;

                        const int li = l - 1; // 0..s-1
                        const std::size_t mbase =
                            (static_cast<std::size_t>(cid) * static_cast<std::size_t>(s) + static_cast<std::size_t>(li))
                            * kMaskWords;
                        const int word = (code >> 5);
                        const int bit = (code & 31);
                        g_mask[mbase + static_cast<std::size_t>(word)] |= (1u << static_cast<std::uint32_t>(bit));
                    }

                    // Tss: Σ a_p * x   (T is Hs×d, column-major)
                    for (int t = 0; t < used; ++t) {
                        const int p = pidx[t];
                        const auto at = static_cast<double>(alpha[t]);
                        double* row = T_loc_t + static_cast<std::size_t>(p) * static_cast<std::size_t>(d);
                        for (int r = 0; r < d; ++r) {
                            row[static_cast<std::size_t>(r)] += at * static_cast<double>(x[r]);
                        }
                    }

                    // Gss: Σ a_p * a_q (symmetric, packed upper)
                    for (int a = 0; a < used; ++a) {
                        const int pa = pidx[a];
                        const auto aa = static_cast<double>(alpha[a]);
                        for (int b = a; b < used; ++b) {
                            const int pb = pidx[b];
                            const int i = std::min(pa, pb);
                            const int j2 = std::max(pa, pb);
                            G_packed[PackUpper(i, j2)] += aa * static_cast<double>(alpha[b]);
                        }
                    }
                }
            }
            if (oob_code.load(std::memory_order_relaxed)) {
                return false;
            }
            if (profile) {
                prof_accum_sec += (omp_get_wtime() - t_accum);
            }
        }

        // Cast stats to float (double accumulation -> float system), matching the in-memory LS numeric regime.
        std::vector<double> Gss_packed_total(packed_phase_a, 0.0);
        std::vector<double> Tss_d(td_size_phase_a, 0.0);

        const double t_reduce_stats = profile ? omp_get_wtime() : 0.0;
#pragma omp parallel for default(none) shared(Gss_packed_total, Gss_packed_all) firstprivate(packed_phase_a, nth_phase_a) schedule(static)
        for (std::int64_t k64 = 0; k64 < static_cast<std::int64_t>(packed_phase_a); ++k64) {
            const auto k = static_cast<std::size_t>(k64);
            double sum = 0.0;
            for (int tid = 0; tid < nth_phase_a; ++tid) {
                sum += Gss_packed_all[static_cast<std::size_t>(tid) * packed_phase_a + k];
            }
            Gss_packed_total[k] = sum;
        }

#pragma omp parallel for default(none) shared(Tss_d, Tss_all_t) firstprivate(td_size_phase_a, nth_phase_a, Hs, d) schedule(static)
        for (std::int64_t k64 = 0; k64 < static_cast<std::int64_t>(td_size_phase_a); ++k64) {
            const auto k = static_cast<std::size_t>(k64);
            const int r = static_cast<int>(k % static_cast<std::size_t>(Hs));
            const int c = static_cast<int>(k / static_cast<std::size_t>(Hs));
            const std::size_t src_off =
                static_cast<std::size_t>(r) * static_cast<std::size_t>(d) + static_cast<std::size_t>(c);
            double sum = 0.0;
            for (int tid = 0; tid < nth_phase_a; ++tid) {
                sum += Tss_all_t[static_cast<std::size_t>(tid) * td_size_phase_a + src_off];
            }
            Tss_d[static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * static_cast<std::size_t>(Hs)] = sum;
        }
        if (profile) {
            prof_reduce_stats_sec += (omp_get_wtime() - t_reduce_stats);
        }

        const double t_cast_stats = profile ? omp_get_wtime() : 0.0;
        std::vector<float> Gss_f(static_cast<std::size_t>(Hs) * static_cast<std::size_t>(Hs), 0.0f);
        std::vector<float> Tss_f(static_cast<std::size_t>(Hs) * static_cast<std::size_t>(d), 0.0f);
        for (int j = 0; j < Hs; ++j) {
            for (int i = 0; i <= j; ++i) {
                const auto v = static_cast<float>(Gss_packed_total[PackUpper(i, j)]);
                Gcm(Gss_f, Hs, i, j) = v;
                Gcm(Gss_f, Hs, j, i) = v;
            }
        }
        for (int c = 0; c < d; ++c) {
            for (int r = 0; r < Hs; ++r) {
                Tcm(Tss_f, Hs, d, r, c) = static_cast<float>(Td(Tss_d, Hs, d, r, c));
            }
        }
        if (profile) {
            prof_cast_stats_sec += (omp_get_wtime() - t_cast_stats);
        }
        std::vector<float> den0_f(static_cast<std::size_t>(h0), 0.0f);
        std::vector<float> t0_f(static_cast<std::size_t>(h0) * static_cast<std::size_t>(d), 0.0f);
        for (int cid = 0; cid < h0; ++cid) {
            const int tid = std::min(nth_phase_a - 1, cid / cids_per_part);
            const std::size_t cbase =
                static_cast<std::size_t>(tid) * static_cast<std::size_t>(cids_per_part) +
                static_cast<std::size_t>(cid - tid * cids_per_part);
            den0_f[static_cast<std::size_t>(cid)] = static_cast<float>(den0_part[cbase]);

            for (int r = 0; r < d; ++r) {
                t0_f[static_cast<std::size_t>(cid) * static_cast<std::size_t>(d) + static_cast<std::size_t>(r)] =
                    static_cast<float>(t0_part[cbase * static_cast<std::size_t>(d) + static_cast<std::size_t>(r)]);
            }
        }

        // Release large double buffers early to reduce peak memory.
        std::vector<double>().swap(Tss_d);
        den0_part_u.reset();
        t0_part_u.reset();
        Gss_packed_all_u.reset();
        Tss_all_t_u.reset();
        std::vector<double>().swap(Gss_packed_total);

        // ---- Phase B: Schur assembly + Cholesky solve (bump schedule) ----
        std::vector<float> W_small; // Hs×d, column-major (solution V)
        float lambda_used = 0.0f;

        const int nth = std::max(1, omp_get_max_threads());
        const std::size_t packed = static_cast<std::size_t>(Hs) * static_cast<std::size_t>(Hs + 1) / 2u;
        const std::size_t td_size = static_cast<std::size_t>(Hs) * static_cast<std::size_t>(d);

        std::vector<double> dG_all(packed * static_cast<std::size_t>(nth), 0.0);
        // NOTE: store dT in transposed layout per thread (matching Phase A's Tss_all_t):
        //   dT_t[p, r] is stored as contiguous d for each p (row-major blocks).
        // This makes the hot-loop update "add scaled t0c[r]" contiguous in memory.
        std::vector<double> dT_all_t(td_size * static_cast<std::size_t>(nth), 0.0);
        std::vector<double> dG_total(packed, 0.0);
        std::vector<double> dT_total_t(td_size, 0.0);

        // Match the full in-memory LS bump policy: if any root cluster has zero diagonal mass (den0==0),
        // the full (h0+Hs) normal matrix is singular and Spotrf would fail at λ=0.
        // In that case, skip λ=0 and start from λ=kEps.
        bool needs_root_bump = false;
        for (int cid = 0; cid < h0; ++cid) {
            if (den0_f[static_cast<std::size_t>(cid)] <= 0.0f) {
                needs_root_bump = true;
                break;
            }
        }

        for (int attempt = 0; attempt < 11; ++attempt) {
            float lambda = 0.0f;
            if (attempt > 0) {
                lambda = kEps;
                for (int t = 1; t < attempt; ++t) {
                    lambda *= 10.0f;
                }
            }

            if (attempt == 0 && needs_root_bump) {
                continue;
            }

            std::fill(dG_all.begin(), dG_all.end(), 0.0);
            std::fill(dT_all_t.begin(), dT_all_t.end(), 0.0);

            // Parallel Schur correction accumulation.
            // We accumulate:
            //   dG += Σ_c (g_c g_c^T)/(den_c+λ)   (upper packed)
            //   dT += Σ_c (g_c t0_c^T)/(den_c+λ) (full)
            // Then:
            //   G_work = Gss + λI - dG
            //   T_work = Tss - dT
            const double t_schur_accum = profile ? omp_get_wtime() : 0.0;
#pragma omp parallel default(none) shared(dG_all, dT_all_t, den0_f, t0_f, gt, g_mask, g_hist) firstprivate(packed, td_size, h0, Hs, d, s, kMaskWords, lambda, PackUpper)
            {
                const int tid = omp_get_thread_num();
                double* dG = dG_all.data() + static_cast<std::size_t>(tid) * packed;
                double* dT_t = dT_all_t.data() + static_cast<std::size_t>(tid) * td_size;

                // Per-thread scratch (m<=16, each small layer <=256).
                int nnz_layer[16];
                int p_layer[16][256];
                float g_layer[16][256];
                int p_all[4096];
                float g_allv[4096];

#pragma omp for schedule(static)
                for (int cid = 0; cid < h0; ++cid) {
                    const float denom = den0_f[static_cast<std::size_t>(cid)] + lambda;
                    if (denom <= 0.0f) {
                        continue;
                    }
                    const double inv = 1.0 / static_cast<double>(denom);

                    const std::size_t gbase = static_cast<std::size_t>(cid) * static_cast<std::size_t>(Hs);

                    int nnz_total = 0;
                    for (int li = 0; li < s; ++li) {
                        nnz_layer[li] = 0;
                        const int l = li + 1;
                        const int off = gt.offsets_small[static_cast<std::size_t>(l)];
                        const std::size_t mbase =
                            (static_cast<std::size_t>(cid) * static_cast<std::size_t>(s) + static_cast<std::size_t>(li))
                            * kMaskWords;
                        for (int w = 0; w < kMaskWords; ++w) {
                            std::uint32_t word = g_mask[mbase + static_cast<std::size_t>(w)];
                            while (word) {
                                const int b = Ctz32NonZero(word);
                                const int code = (w << 5) + b; // w*32 + b
                                const int p = off + code;
                                const float gv = g_hist[gbase + static_cast<std::size_t>(p)];
                                if (gv != 0.0f) {
                                    const int k = nnz_layer[li]++;
                                    p_layer[li][k] = p;
                                    g_layer[li][k] = gv;
                                    p_all[nnz_total] = p;
                                    g_allv[nnz_total] = gv;
                                    ++nnz_total;
                                }
                                word &= (word - 1u);
                            }
                        }
                    }
                    if (nnz_total == 0) {
                        continue;
                    }

                    // RHS correction dT (transposed per-thread layout):
                    //   for each p in nnz(g): dT[p,:] += (g[p]/denom) * t0c[:]
                    const float* t0c = t0_f.data() + static_cast<std::size_t>(cid) * static_cast<std::size_t>(d);
                    for (int k = 0; k < nnz_total; ++k) {
                        const int p = p_all[k];
                        const double alpha = static_cast<double>(g_allv[k]) * inv;
                        double* row = dT_t + static_cast<std::size_t>(p) * static_cast<std::size_t>(d);
                        for (int r = 0; r < d; ++r) {
                            row[static_cast<std::size_t>(r)] += alpha * static_cast<double>(t0c[r]);
                        }
                    }

                    // Gram correction dG (upper packed). Exploit block structure by layer.
                    for (int l1 = 0; l1 < s; ++l1) {
                        const int n1 = nnz_layer[l1];
                        if (n1 == 0) continue;

                        // Diagonal block (l1,l1): i<=j already because we enumerate codes in ascending order.
                        for (int a = 0; a < n1; ++a) {
                            const int i = p_layer[l1][a];
                            const float gi = g_layer[l1][a];
                            for (int b = a; b < n1; ++b) {
                                const int j = p_layer[l1][b];
                                const float gj = g_layer[l1][b];
                                dG[PackUpper(i, j)] += static_cast<double>(gi) * static_cast<double>(gj) * inv;
                            }
                        }

                        // Off-diagonal blocks (l1,l2), l2>l1: offsets are increasing -> i<j always.
                        for (int l2 = l1 + 1; l2 < s; ++l2) {
                            const int n2 = nnz_layer[l2];
                            if (n2 == 0) continue;
                            for (int a = 0; a < n1; ++a) {
                                const int i = p_layer[l1][a];
                                const float gi = g_layer[l1][a];
                                for (int b = 0; b < n2; ++b) {
                                    const int j = p_layer[l2][b];
                                    const float gj = g_layer[l2][b];
                                    dG[PackUpper(i, j)] += static_cast<double>(gi) * static_cast<double>(gj) * inv;
                                }
                            }
                        }
                    }
                }
            } // omp parallel
            if (profile) {
                prof_schur_accum_sec += (omp_get_wtime() - t_schur_accum);
            }

            // Reduce thread-local accumulators.
            const double t_schur_reduce = profile ? omp_get_wtime() : 0.0;
            std::fill(dG_total.begin(), dG_total.end(), 0.0);
            std::fill(dT_total_t.begin(), dT_total_t.end(), 0.0);
            for (int tid = 0; tid < nth; ++tid) {
                const double* srcG = dG_all.data() + static_cast<std::size_t>(tid) * packed;
                const double* srcT_t = dT_all_t.data() + static_cast<std::size_t>(tid) * td_size;
                for (std::size_t k = 0; k < packed; ++k) {
                    dG_total[k] += srcG[k];
                }
                for (std::size_t k = 0; k < td_size; ++k) {
                    dT_total_t[k] += srcT_t[k];
                }
            }
            if (profile) {
                prof_schur_reduce_sec += (omp_get_wtime() - t_schur_reduce);
            }

            // Build Schur system.
            const double t_schur_build = profile ? omp_get_wtime() : 0.0;
            // Assemble in double to reduce cancellation (Gss and Schur correction are same-order terms),
            // then cast to float for the LAPACK float Cholesky solve to match the in-memory LS numeric regime.
            std::vector<float> G_work(static_cast<std::size_t>(Hs) * static_cast<std::size_t>(Hs), 0.0f);
            std::vector<float> T_work(static_cast<std::size_t>(Hs) * static_cast<std::size_t>(d), 0.0f);

            // Upper triangle only (SpotrfU reads upper).
            for (int j = 0; j < Hs; ++j) {
                for (int i = 0; i <= j; ++i) {
                    const auto base = static_cast<double>(Gcm(Gss_f, Hs, i, j));
                    const double corr = dG_total[PackUpper(i, j)];
                    double v = base - corr;
                    if (i == j) v += static_cast<double>(lambda);
                    Gcm(G_work, Hs, i, j) = static_cast<float>(v);
                }
            }

            // RHS: T = Ts - dT.
            for (int r = 0; r < d; ++r) {
                for (int p = 0; p < Hs; ++p) {
                    const std::size_t dst_cm = static_cast<std::size_t>(p) + static_cast<std::size_t>(r) * static_cast<
                        std::size_t>(Hs);
                    const std::size_t src_t = static_cast<std::size_t>(p) * static_cast<std::size_t>(d) + static_cast<
                        std::size_t>(r);
                    T_work[dst_cm] = static_cast<float>(static_cast<double>(Tss_f[dst_cm]) - dT_total_t[src_t]);
                }
            }
            if (profile) {
                prof_schur_build_sec += (omp_get_wtime() - t_schur_build);
            }

            const double t_schur_solve = profile ? omp_get_wtime() : 0.0;
            int info = lapack::SpotrfU(Hs, G_work.data(), Hs);
            if (info != 0) {
                continue; // try larger λ
            }
            info = lapack::SpotrsU(Hs, d, G_work.data(), Hs, T_work.data(), Hs);
            if (info != 0) {
                if (err) {
                    *err = "UpdateCRootFromTrainBasicStreamingExactLS: potrs failed (info=" + std::to_string(info) +
                        ").";
                }
                return false;
            }
            if (profile) {
                prof_schur_solve_sec += (omp_get_wtime() - t_schur_solve);
            }

            lambda_used = lambda;
            W_small = std::move(T_work);
            break;
        }

        if (W_small.empty()) {
            if (err) *err = "UpdateCRootFromTrainBasicStreamingExactLS: Schur solve failed for all bumps.";
            return false;
        }

        // ---- Phase C: write back small layers, recover root layer ----
        const double t_write_small = profile ? omp_get_wtime() : 0.0;
        for (int l = 1; l < m; ++l) {
            const int hl = cfg.model.h_vec[static_cast<std::size_t>(l)];
            ColMajorMatrix<float> book(d, hl);
            for (int code = 0; code < hl; ++code) {
                const int p = gt.offsets_small[static_cast<std::size_t>(l)] + code;
                float* dst = book.Col(code);
                for (int r = 0; r < d; ++r) {
                    dst[r] = W_small[static_cast<std::size_t>(p) + static_cast<std::size_t>(r) * static_cast<
                        std::size_t>(Hs)];
                }
            }
            C_root_inout->books[static_cast<std::size_t>(l)] = std::move(book);
        }
        if (profile) {
            prof_write_small_sec += (omp_get_wtime() - t_write_small);
        }

        // Recover root layer:
        //   c0_c = (t0_c - Σ_p g_c[p] * V_p) / (den_c + λ)
        // Optimization note: W_small is Hs×d column-major. For a sparse g row, computing g^T*W_small naively
        // (loop over r then gather W_small[p + r*Hs]) is cache-unfriendly due to large stride Hs between columns.
        // We precompute Vt = W_small^T as a d×Hs column-major matrix so each selected column is contiguous in r.
        std::vector<float> Vt(static_cast<std::size_t>(d) * static_cast<std::size_t>(Hs));
        for (int p = 0; p < Hs; ++p) {
            float* dst = Vt.data() + static_cast<std::size_t>(p) * static_cast<std::size_t>(d);
            for (int r = 0; r < d; ++r) {
                dst[static_cast<std::size_t>(r)] =
                    W_small[static_cast<std::size_t>(p) + static_cast<std::size_t>(r) * static_cast<std::size_t>(Hs)];
            }
        }

        ColMajorMatrix<float> book0(d, h0);
        const double t_recover_root = profile ? omp_get_wtime() : 0.0;
#pragma omp parallel default(none) shared(book0, den0_f, C_root_inout, gt, g_mask, g_hist, t0_f, Vt) firstprivate(h0, Hs, d, s, kMaskWords, lambda_used)
        {
            int p_all[4096];
            float g_allv[4096];

#pragma omp for schedule(static)
            for (int cid = 0; cid < h0; ++cid) {
                const float den0v = den0_f[static_cast<std::size_t>(cid)];
                const float denom = den0v + lambda_used;
                float* c0 = book0.Col(cid);
                if (den0v <= 0.0f) {
                    // This root cluster is unconstrained by data (den0==0). Writing a zero vector here
                    // can permanently kill the cluster downstream (large-root beam search skips alpha==0).
                    // Preserve the previous root center instead.
                    const float* prev = C_root_inout->books[0].Col(cid);
                    std::memcpy(c0, prev, static_cast<std::size_t>(d) * sizeof(float));
                    continue;
                }
                if (denom <= 0.0f) {
                    // Extremely defensive: should not happen with lambda_used>=0, but keep sane behavior.
                    const float* prev = C_root_inout->books[0].Col(cid);
                    std::memcpy(c0, prev, static_cast<std::size_t>(d) * sizeof(float));
                    continue;
                }
                const double inv = 1.0 / static_cast<double>(denom);

                const std::size_t gbase = static_cast<std::size_t>(cid) * static_cast<std::size_t>(Hs);
                int nnz_total = 0;
                for (int li = 0; li < s; ++li) {
                    const int l = li + 1;
                    const int off = gt.offsets_small[static_cast<std::size_t>(l)];
                    const std::size_t mbase =
                        (static_cast<std::size_t>(cid) * static_cast<std::size_t>(s) + static_cast<std::size_t>(li)) *
                        kMaskWords;
                    for (int w = 0; w < kMaskWords; ++w) {
                        std::uint32_t word = g_mask[mbase + static_cast<std::size_t>(w)];
                        while (word) {
                            const int b = Ctz32NonZero(word);
                            const int code = (w << 5) + b;
                            const int p = off + code;
                            const float gv = g_hist[gbase + static_cast<std::size_t>(p)];
                            if (gv != 0.0f) {
                                p_all[nnz_total] = p;
                                g_allv[nnz_total] = gv;
                                ++nnz_total;
                            }
                            word &= (word - 1u);
                        }
                    }
                }

                const float* t0c = t0_f.data() + static_cast<std::size_t>(cid) * static_cast<std::size_t>(d);

                // Start from t0 and subtract Σ g[p] * V[p,:]. Using Vt makes each V[:,p] contiguous in r.
                for (int r = 0; r < d; ++r) {
                    c0[r] = t0c[r];
                }
                for (int k = 0; k < nnz_total; ++k) {
                    const int p = p_all[k];
                    const float g = g_allv[k];
                    const float* col = Vt.data() + static_cast<std::size_t>(p) * static_cast<std::size_t>(d);
                    for (int r = 0; r < d; ++r) {
                        c0[r] -= g * col[static_cast<std::size_t>(r)];
                    }
                }
                for (int r = 0; r < d; ++r) {
                    c0[r] = c0[r] * inv;
                }
            }
        } // omp parallel
        if (profile) {
            prof_recover_root_sec += (omp_get_wtime() - t_recover_root);
        }

        C_root_inout->books[0] = std::move(book0);
        if (profile) {
            const auto Fmt = [](double v, int prec)
            {
                std::ostringstream oss;
                oss.setf(std::ios::fixed);
                oss << std::setprecision(prec) << v;
                return oss.str();
            };
            const double sum =
                prof_read_basic_sec +
                prof_read_x_sec +
                prof_convert_rotate_sec +
                prof_partition_sec +
                prof_accum_sec +
                prof_reduce_stats_sec +
                prof_cast_stats_sec +
                prof_schur_accum_sec +
                prof_schur_reduce_sec +
                prof_schur_build_sec +
                prof_schur_solve_sec +
                prof_write_small_sec +
                prof_recover_root_sec;
            const double total = profile ? (omp_get_wtime() - prof_total_t0) : 0.0;
            LogInfo("Update C_root timing breakdown (profile, wall): read_basic=" +
                Fmt(prof_read_basic_sec, 3) +
                " read_x=" + Fmt(prof_read_x_sec, 3) +
                " convert_rotate=" + Fmt(prof_convert_rotate_sec, 3) +
                " partition=" + Fmt(prof_partition_sec, 3) +
                " accum=" + Fmt(prof_accum_sec, 3) +
                " reduce_stats=" + Fmt(prof_reduce_stats_sec, 3) +
                " cast_stats=" + Fmt(prof_cast_stats_sec, 3) +
                " schur_accum=" + Fmt(prof_schur_accum_sec, 3) +
                " schur_reduce=" + Fmt(prof_schur_reduce_sec, 3) +
                " schur_build=" + Fmt(prof_schur_build_sec, 3) +
                " schur_solve=" + Fmt(prof_schur_solve_sec, 3) +
                " write_small=" + Fmt(prof_write_small_sec, 3) +
                " recover_root=" + Fmt(prof_recover_root_sec, 3) +
                " lambda=" + Fmt(static_cast<double>(lambda_used), 9) +
                " sum=" + Fmt(sum, 3) +
                " total=" + Fmt(total, 3));
        }

        // Optional: compute training MSE after updating C_root using stored (cluster_id, codes_small, a).
        if (out_mse_after_update) {
            const stlq::Timer mse_wall;
            double sse_sum = 0.0;
            std::uint64_t n_seen = 0;

            for (std::uint64_t start = 0; start < ntrain; start += block) {
                const std::uint32_t count =
                    static_cast<std::uint32_t>(std::min<std::uint64_t>(block, ntrain - start));
                if (!basic.ReadClusterIdBlock(start, count, &cluster_id, err)) {
                    return false;
                }
                if (!basic.ReadCodesBlock(start, count, &B_small, err)) {
                    return false;
                }
                if (!basic.ReadCoeffsBlock(start, count, &a_all, err)) {
                    return false;
                }
                if (!read_train_block_rotated_exact(start, count)) {
                    return false;
                }

                double sse_block = 0.0;
#pragma omp parallel default(none) shared(cluster_id, a_all, Xrot, C_root_inout, B_small) firstprivate(count, d, m) reduction(+:sse_block)
                {
                    std::vector<float> recon(static_cast<std::size_t>(d), 0.0f);
#pragma omp for schedule(static)
                    for (std::int64_t j64 = 0; j64 < static_cast<std::int64_t>(count); ++j64) {
                        const auto j = static_cast<std::uint32_t>(j64);
                        const int cid = static_cast<int>(cluster_id[static_cast<std::size_t>(j)]);
                        const float a0 = a_all(0, static_cast<int>(j));
                        const float* x = Xrot.Col(static_cast<int>(j));

                        const float* c0 = C_root_inout->books[0].Col(cid);
                        for (int r = 0; r < d; ++r) {
                            recon[static_cast<std::size_t>(r)] = a0 * c0[r];
                        }
                        for (int l = 1; l < m; ++l) {
                            const int code = static_cast<int>(B_small(l - 1, static_cast<int>(j)));
                            const float al = a_all(l, static_cast<int>(j));
                            const float* center = C_root_inout->books[static_cast<std::size_t>(l)].Col(code);
                            for (int r = 0; r < d; ++r) {
                                recon[static_cast<std::size_t>(r)] += al * center[r];
                            }
                        }

                        double e = 0.0;
                        for (int r = 0; r < d; ++r) {
                            const double diff =
                                static_cast<double>(x[r]) - static_cast<double>(recon[static_cast<std::size_t>(r)]);
                            e += diff * diff;
                        }
                        sse_block += e;
                    }
                }
                sse_sum += sse_block;
                n_seen += static_cast<std::uint64_t>(count);
            }
            *out_mse_after_update = (n_seen > 0)
                                        ? static_cast<float>(sse_sum / static_cast<double>(n_seen))
                                        : 0.0f;
            if (out_mse_wall_sec) {
                *out_mse_wall_sec = mse_wall.ElapsedSeconds();
            }
        }

        BumpCodebookBuildTag(C_root_inout);
        return true;
    }


    bool UpdateCOneFromTrainLinkageListStreaming(const Config& cfg,
                                                 const TrainResult& train,
                                                 const io::BaseListReader& train_list,
                                                 const io::IvfListsReader& ivf,
                                                 const io::LinkageListReader& linkage_list,
                                                 const io::DatasetVectorReader* fallback_reader,
                                                 bool allow_random_fallback,
                                                 StreamKernelProvider* kernels,
                                                 CodebookPack* C_one_inout,
                                                 std::string* err) {
        (void)allow_random_fallback;
        if (!kernels) {
            if (err) *err = "UpdateCOneFromTrainLinkageListStreaming: kernels is null.";
            return false;
        }
        if (!C_one_inout || C_one_inout->books.empty()) {
            if (err) *err = "UpdateCOneFromTrainLinkageListStreaming: empty C_one.";
            return false;
        }
        const int d = train_list.meta().d;
        const int m = cfg.model.m;
        const int m_codes = std::max(0, m - 1);
        if (d <= 0 || m <= 1 || m_codes != linkage_list.m_codes()) {
            if (err) *err = "UpdateCOneFromTrainLinkageListStreaming: invalid d/m/m_codes.";
            return false;
        }

        // Snapshot old C_one for parent reconstruction (must match the linkage build step).
        const CodebookPack C_one_old = *C_one_inout;

        io::LinkageListThreadReader linkage_thr;
        if (!linkage_thr.OpenFrom(linkage_list, err)) {
            return false;
        }
        io::BaseListThreadReader train_thr;
        if (!train_thr.OpenFrom(train_list, err)) {
            return false;
        }
        io::IvfListsThreadReader ivf_thr;
        if (!ivf_thr.OpenFrom(ivf, err)) {
            return false;
        }

        const bool has_raw_u8 = train_list.HasRawU8();
        const bool has_raw_f32 = train_list.HasRawF32();
        // IMPORTANT: do NOT mix dtypes. Prefer a single raw dtype for this scan.
        // - If an f32 fallback reader is provided, treat as f32-only (never consume raw_u8.bin).
        // - If a u8 fallback reader is provided, treat as u8-only (never consume raw_f32.bin).
        // - Otherwise, use the on-disk raw span dtype when available.
        const bool is_f32_run = ReaderCanReadF32(fallback_reader) || (has_raw_f32 && !has_raw_u8);
        const bool is_u8_run = ReaderCanReadU8(fallback_reader) || (has_raw_u8 && !has_raw_f32);
        if (!is_f32_run && !is_u8_run) {
            if (err)
                *err =
                    "UpdateCOneFromTrainLinkageListStreaming: no raw vectors available (need raw_u8/raw_f32 span or fallback reader).";
            return false;
        }
        if (is_f32_run && is_u8_run && !fallback_reader) {
            if (err)
                *err =
                    "UpdateCOneFromTrainLinkageListStreaming: ambiguous raw dtype (both raw_u8 and raw_f32 present).";
            return false;
        }
        int code0_width_bytes = 0;
        if (!ValidateLinkageListCodeLayout(linkage_list,
                                           /*allow_wide_code0=*/false,
                                           "UpdateCOneFromTrainLinkageListStreaming",
                                           &code0_width_bytes,
                                           err)) {
            return false;
        }

        auto h_vec_one = cfg.model.h_vec;
        if (!h_vec_one.empty()) {
            h_vec_one[0] = cfg.model.h0_one;
        }

        // Flatten offsets for all layers including l=0.
        std::vector<int> offsets_all(static_cast<std::size_t>(m), 0);
        int H = 0;
        for (int l = 0; l < m; ++l) {
            offsets_all[static_cast<std::size_t>(l)] = H;
            H += std::max(1, h_vec_one[static_cast<std::size_t>(l)]);
        }
        if (H <= 0) {
            if (err) *err = "UpdateCOneFromTrainLinkageListStreaming: invalid flattened H.";
            return false;
        }

        auto PackUpper = [](int i, int j) -> std::size_t
        {
            // i <= j
            return static_cast<std::size_t>(j) * static_cast<std::size_t>(j + 1) / 2u + static_cast<std::size_t>(i);
        };
        const std::size_t packed_size = static_cast<std::size_t>(H) * static_cast<std::size_t>(H + 1) / 2u;
        const std::size_t td_size_t = static_cast<std::size_t>(H) * static_cast<std::size_t>(d); // transposed

        const int omp_max = OmpMaxThreads();
        int shards = cfg.runtime.c_one_update_shards;
        if (shards <= 0) {
            shards = std::min(omp_max, 16);
        }
        shards = std::clamp(shards, 1, omp_max);

        const bool can_parallel_accum = (shards > 1 && omp_max > 1);
        if (can_parallel_accum) {
            // This stage is bandwidth-bound (scatter-add into dense normal equations). Use a sharded
            // accumulator to bound memory: O(shards * H^2) instead of O(omp_threads * H^2).
            // Also keep BLAS single-threaded inside the OpenMP region to avoid oversubscription.
            ScopedBlasThreads blas_scope(1);
            ScopedOmpThreads omp_scope(shards);

            const std::size_t G_packed_all_size = packed_size * static_cast<std::size_t>(shards);
            auto G_packed_all_u = std::unique_ptr<double[]>(new double[G_packed_all_size]);
            double* G_packed_all = G_packed_all_u.get();
            const std::size_t T_all_t_size = td_size_t * static_cast<std::size_t>(shards);
            auto T_all_t_u = std::unique_ptr<double[]>(new double[T_all_t_size]);
            double* T_all_t = T_all_t_u.get();

#pragma omp parallel default(none) shared(G_packed_all, T_all_t, linkage_list, train_list, ivf, cfg, train, fallback_reader, offsets_all, C_one_old) firstprivate(PackUpper, packed_size, td_size_t, d, m, m_codes, has_raw_u8, has_raw_f32, is_f32_run, is_u8_run, code0_width_bytes)
            {
                const int tid = omp_get_thread_num();
                double* G_packed = G_packed_all + static_cast<std::size_t>(tid) * packed_size;
                double* T_t = T_all_t + static_cast<std::size_t>(tid) * td_size_t;
                std::fill(G_packed, G_packed + packed_size, 0.0);
                std::fill(T_t, T_t + td_size_t, 0.0);

                auto Gd = [&](int r, int c) -> double& {
                    return G_packed[PackUpper(r, c)];
                };
                auto Td = [&](int p, int r) -> double& {
                    return T_t[static_cast<std::size_t>(p) * static_cast<std::size_t>(d) + static_cast<std::size_t>(r)];
                };

                // Per-thread IO readers + CPU kernels (thread-safe).
                io::LinkageListThreadReader linkage_thr_tls;
                io::BaseListThreadReader train_thr_tls;
                io::IvfListsThreadReader ivf_thr_tls;
                std::string open_err;
                if (!linkage_thr_tls.OpenFrom(linkage_list, &open_err) ||
                    !train_thr_tls.OpenFrom(train_list, &open_err) ||
                    !ivf_thr_tls.OpenFrom(ivf, &open_err)) {
                    // Leave local slices zero; reduction will still work and the solve will likely fail,
                    // but we avoid throwing from inside OMP.
                }

                CpuStreamKernels kernels_tls;
                StreamKernelProvider* kernels_local = &kernels_tls;

                std::vector<std::uint32_t> ids;
                std::vector<std::pair<std::uint32_t, int>> gid_pos;
                std::vector<std::uint32_t> real_ids;
                std::vector<std::uint32_t> parent;
                std::vector<std::uint32_t> depth_offsets;
                std::vector<std::uint8_t> codes_bytes;
                std::vector<float> coeffs_small;
                std::vector<std::uint8_t> code0_one_bytes;
                std::vector<float> a0;
                std::vector<std::uint8_t> virt_codes_bytes;
                std::vector<float> virt_coeffs;
                std::vector<float> virt_a0;

                ColMajorMatrix<std::uint8_t> x_u8;
                ColMajorMatrix<float> x_f32;
                ColMajorMatrix<float> Xrot;
                ColMajorMatrix<float> R_full; // d×n_real, depth-order
                ColMajorMatrix<float> R_virt; // d×n_virt, virtual roots reconstruction
                std::vector<float> tmp_self(static_cast<std::size_t>(d), 0.0f);
                std::vector<float> x_target(static_cast<std::size_t>(d), 0.0f);

                int idx_flat[32];
                float alpha[32];

                // G/T are accumulated per thread and reduced in fixed shard order below.
                // Keep the cluster-to-shard assignment fixed as well, otherwise floating-point
                // grouping depends on OpenMP timing and identical seeds can diverge across rounds.
#pragma omp for schedule(static)
                for (int cid = 0; cid < ivf.nlist(); ++cid) {
                    const double w_cluster =
                    (!train.is_bad_cluster.empty() &&
                        cid >= 0 && cid < static_cast<int>(train.is_bad_cluster.size()) &&
                        train.is_bad_cluster[static_cast<std::size_t>(cid)])
                        ? static_cast<double>(cfg.virtual_cfg.alpha_bad)
                        : 1.0;
                    std::string local_err;
                    if (!ivf_thr_tls.ReadList(ivf, cid, &ids, &local_err)) {
                        continue;
                    }
                    const int n_real = static_cast<int>(ids.size());
                    if (n_real <= 0) {
                        continue;
                    }
                    {
                        std::string ignored;
                        if (!linkage_thr_tls.ReadCluster(cid, &real_ids, &parent, &depth_offsets,
                                                         &codes_bytes, &coeffs_small,
                                                         &code0_one_bytes, &a0,
                                                         &virt_codes_bytes, &virt_coeffs, &virt_a0,
                                                         &ignored)) {
                            continue;
                        }
                    }
                    if (static_cast<int>(real_ids.size()) != n_real || static_cast<int>(parent.size()) != n_real) {
                        continue;
                    }
                    if (depth_offsets.size() < 2) {
                        continue;
                    }
                    const int depth1_start = static_cast<int>(depth_offsets[1]);
                    if (depth1_start >= n_real) {
                        continue;
                    }

                    // gid->pos map for list-order alignment.
                    gid_pos.clear();
                    gid_pos.reserve(static_cast<std::size_t>(n_real));
                    for (int i = 0; i < n_real; ++i) {
                        gid_pos.emplace_back(ids[static_cast<std::size_t>(i)], i);
                    }
                    std::sort(gid_pos.begin(), gid_pos.end(),
                              [](const auto& a, const auto& b) { return a.first < b.first; });

                    // Read raw vectors in list-order and rotate.
                    if (is_f32_run) {
                        if (has_raw_f32) {
                            if (!train_thr_tls.ReadRawF32Span(ivf.Offset(cid), static_cast<std::uint32_t>(n_real),
                                                              &x_f32, &local_err)) {
                                LogWarn(
                                    "UpdateCOne[cid=" + std::to_string(cid) + "]: raw f32 read failed: " + local_err);
                                continue;
                            }
                        }
                        else if (ReaderCanReadF32(fallback_reader)) {
                            if (!io::ReadDatasetVectorByIdsF32(*fallback_reader, ids, &x_f32, &local_err)) {
                                LogWarn(
                                    "UpdateCOne[cid=" + std::to_string(cid) + "]: fvecs fallback read failed: " +
                                    local_err);
                                continue;
                            }
                        }
                        else if (ReaderCanReadF32(fallback_reader)) {
                            if (!io::ReadDatasetVectorByIdsF32(*fallback_reader, ids, &x_f32, &local_err)) {
                                LogWarn(
                                    "UpdateCOne[cid=" + std::to_string(cid) + "]: fbin fallback read failed: " +
                                    local_err);
                                continue;
                            }
                        }
                        else {
                            LogWarn("UpdateCOne[cid=" + std::to_string(cid) + "]: no f32 vector source available.");
                            continue;
                        }
                        Xrot.rows = train.R.rows;
                        Xrot.cols = x_f32.cols;
                        Xrot.data.resize(static_cast<std::size_t>(Xrot.rows) * static_cast<std::size_t>(Xrot.cols));
                        kernels_local->Gemm(false, false, 1.0f, train.R, x_f32, 0.0f, &Xrot);
                    }
                    else if (is_u8_run) {
                        if (has_raw_u8) {
                            if (!train_thr_tls.ReadRawU8Span(ivf.Offset(cid), static_cast<std::uint32_t>(n_real), &x_u8,
                                                             &local_err)) {
                                LogWarn(
                                    "UpdateCOne[cid=" + std::to_string(cid) + "]: raw u8 read failed: " + local_err);
                                continue;
                            }
                        }
                        else if (ReaderCanReadU8(fallback_reader)) {
                            if (!io::ReadDatasetVectorByIdsU8(*fallback_reader, ids, &x_u8, &local_err)) {
                                LogWarn(
                                    "UpdateCOne[cid=" + std::to_string(cid) + "]: bvecs fallback read failed: " +
                                    local_err);
                                continue;
                            }
                        }
                        else {
                            // Neither raw stream nor fallback file available — configuration error.
                            LogWarn("UpdateCOne[cid=" + std::to_string(cid) + "]: no u8 vector source available.");
                            continue;
                        }
                        kernels_local->ConvertU8ToF32AndRotate(x_u8, train.R, &Xrot);
                    }
                    else {
                        LogWarn(
                            "UpdateCOne[cid=" + std::to_string(cid) +
                            "]: unknown vector dtype (neither f32 nor u8 run).");
                        continue;
                    }

                    // Reconstruct R_full in depth-order using old codebooks (for parent reconstruction only).
                    R_full.rows = d;
                    R_full.cols = n_real;
                    R_full.data.resize(static_cast<std::size_t>(d) * static_cast<std::size_t>(n_real));
                    const std::uint8_t* codes_u8 = codes_bytes.data();
                    const std::uint8_t* code0_one_u8 = code0_one_bytes.data();

                    const int n_virt = (m_codes > 0) ? static_cast<int>(virt_coeffs.size()) / m_codes : 0;
                    R_virt.rows = d;
                    R_virt.cols = n_virt;
                    R_virt.data.resize(static_cast<std::size_t>(d) * static_cast<std::size_t>(std::max(0, n_virt)));
                    const std::uint8_t* vcodes_u8 = virt_codes_bytes.data();
                    for (int v = 0; v < n_virt; ++v) {
                        const std::uint8_t* cu8 = vcodes_u8 + static_cast<std::size_t>(v) * m_codes;
                        const float* a_small = virt_coeffs.data() + static_cast<std::size_t>(v) * m_codes;
                        const float a0v = (v < static_cast<int>(virt_a0.size()))
                                              ? virt_a0[static_cast<std::size_t>(v)]
                                              : 0.0f;
                        AccumulateReconDepth0FromSmallU8(train.C_root, cid, cu8, a_small, a0v, m, d, R_virt.Col(v));
                    }

                    for (int i = 0; i < n_real; ++i) {
                        const std::uint8_t* cu8 = codes_u8 + static_cast<std::size_t>(i) * m_codes;
                        const float* a_small = coeffs_small.data() + static_cast<std::size_t>(i) * m_codes;
                        const float a0i = a0[static_cast<std::size_t>(i)];
                        const std::uint32_t p1 = parent[static_cast<std::size_t>(i)];
                        float* dst = R_full.Col(i);
                        if (p1 == 0) {
                            AccumulateReconDepth0FromSmallU8(train.C_root, cid, cu8, a_small, a0i, m, d, dst);
                        }
                        else {
                            const int code0 = static_cast<int>(code0_one_u8[static_cast<std::size_t>(i)]);
                            AccumulateReconDepthPosFromSmallU8(C_one_old, code0, cu8, a_small, a0i, m, d,
                                                               tmp_self.data());
                            const int p = static_cast<int>(p1 - 1); // local id (virtual-front)
                            if (p < n_virt) {
                                // p >= 0 is guaranteed: p = p1-1 and p1 >= 1.
                                const float* rp = R_virt.Col(p);
                                const float* ts = tmp_self.data();
#pragma omp simd
                                for (int r = 0; r < d; ++r) {
                                    dst[r] = rp[r] + ts[r];
                                }
                            }
                            else {
                                const int pr = p - n_virt; // pr >= 0 guaranteed (p >= n_virt)
                                if (pr < n_real) {
                                    const float* rp = R_full.Col(pr);
                                    const float* ts = tmp_self.data();
#pragma omp simd
                                    for (int r = 0; r < d; ++r) {
                                        dst[r] = rp[r] + ts[r];
                                    }
                                }
                                else {
                                    LogWarn("UpdateCOne[cid=" + std::to_string(cid) + "]: parent pr=" +
                                        std::to_string(pr) + " >= n_real=" + std::to_string(n_real) +
                                        " at node i=" + std::to_string(i) + "; linkage data corrupt.");
                                    std::memcpy(dst, tmp_self.data(), sizeof(float) * static_cast<std::size_t>(d));
                                }
                            }
                        }
                    }

                    // Accumulate normal equations for depth>0 nodes.
                    for (int i = depth1_start; i < n_real; ++i) {
                        const int pos = FindListPos(gid_pos, real_ids[static_cast<std::size_t>(i)]);
                        if (pos < 0 || pos >= n_real) {
                            LogWarn(
                                "UpdateCOne[cid=" + std::to_string(cid) +
                                "]: gid not found in list-order map at node i=" +
                                std::to_string(i) + "; skipping.");
                            continue;
                        }
                        const int p = static_cast<int>(parent[static_cast<std::size_t>(i)] - 1);
                        // p >= 0 guaranteed (parent[i] >= 1 for depth>0). (p-n_virt) >= 0 if p >= n_virt.
                        const float* rp = (p < n_virt)
                                              ? R_virt.Col(p)
                                              : ((p - n_virt < n_real) ? R_full.Col(p - n_virt) : nullptr);
                        if (!rp) {
                            LogWarn("UpdateCOne[cid=" + std::to_string(cid) + "]: parent p=" + std::to_string(p) +
                                " out of range (n_virt=" + std::to_string(n_virt) + " n_real=" + std::to_string(n_real)
                                +
                                ") at node i=" + std::to_string(i) + "; skipping.");
                            continue;
                        }
                        const float* x = Xrot.Col(pos);
                        for (int r = 0; r < d; ++r) {
                            x_target[static_cast<std::size_t>(r)] = x[r] - rp[r];
                        }

                        const int code0 = static_cast<int>(code0_one_u8[static_cast<std::size_t>(i)]);
                        idx_flat[0] = offsets_all[0] + code0;
                        alpha[0] = a0[static_cast<std::size_t>(i)];

                        const std::uint8_t* cu8 = codes_u8 + static_cast<std::size_t>(i) * m_codes;
                        const float* a_small = coeffs_small.data() + static_cast<std::size_t>(i) * m_codes;
                        for (int l = 1; l < m; ++l) {
                            const int idx = l - 1;
                            const int code = static_cast<int>(cu8[static_cast<std::size_t>(idx)]);
                            idx_flat[l] = offsets_all[static_cast<std::size_t>(l)] + code;
                            alpha[l] = a_small[static_cast<std::size_t>(idx)];
                        }

                        for (int u = 0; u < m; ++u) {
                            const int pu = idx_flat[u];
                            const auto au = static_cast<double>(alpha[u]);
                            for (int r = 0; r < d; ++r) {
                                Td(pu, r) += w_cluster * au *
                                    static_cast<double>(x_target[static_cast<std::size_t>(r)]);
                            }
                            for (int v = 0; v <= u; ++v) {
                                const int pv = idx_flat[v];
                                const auto av = static_cast<double>(alpha[v]);
                                const int rr = std::min(pu, pv);
                                const int cc = std::max(pu, pv);
                                Gd(rr, cc) += w_cluster * au * av;
                            }
                        }
                    }
                }
            }

            // Reduce shards into slot 0.
#pragma omp parallel for default(none) shared(G_packed_all) firstprivate(packed_size, shards) schedule(static)
            for (std::int64_t i64 = 0; i64 < static_cast<std::int64_t>(packed_size); ++i64) {
                const auto i = static_cast<std::size_t>(i64);
                double sum = G_packed_all[i];
                for (int s = 1; s < shards; ++s) {
                    sum += G_packed_all[static_cast<std::size_t>(s) * packed_size + i];
                }
                G_packed_all[i] = sum;
            }
#pragma omp parallel for default(none) shared(T_all_t) firstprivate(td_size_t, shards) schedule(static)
            for (std::int64_t i64 = 0; i64 < static_cast<std::int64_t>(td_size_t); ++i64) {
                const auto i = static_cast<std::size_t>(i64);
                double sum = T_all_t[i];
                for (int s = 1; s < shards; ++s) {
                    sum += T_all_t[static_cast<std::size_t>(s) * td_size_t + i];
                }
                T_all_t[i] = sum;
            }

            // Convert to float for LAPACK (upper triangle only).
            std::vector<float> G(static_cast<std::size_t>(H) * static_cast<std::size_t>(H), 0.0f);
            std::vector<float> W(static_cast<std::size_t>(H) * static_cast<std::size_t>(d), 0.0f);
            for (int c = 0; c < H; ++c) {
                for (int r = 0; r <= c; ++r) {
                    G[static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * static_cast<std::size_t>(H)] =
                        static_cast<float>(G_packed_all[PackUpper(r, c)]);
                }
            }
            for (int p = 0; p < H; ++p) {
                const double* src = T_all_t + static_cast<std::size_t>(p) * static_cast<std::size_t>(d);
                for (int r = 0; r < d; ++r) {
                    W[static_cast<std::size_t>(p) + static_cast<std::size_t>(r) * static_cast<std::size_t>(H)] =
                        static_cast<float>(src[static_cast<std::size_t>(r)]);
                }
            }

            float bump = 0.0f;
            bool ok = false;
            for (int attempt = 0; attempt < 8; ++attempt) {
                std::vector<float> Af = G;
                if (bump > 0.0f) {
                    for (int i = 0; i < H; ++i) {
                        Af[static_cast<std::size_t>(i) + static_cast<std::size_t>(i) * static_cast<std::size_t>(H)] +=
                            bump;
                    }
                }
                std::vector<float> Bf = W;
                int info = lapack::SpotrfU(H, Af.data(), H);
                if (info != 0) {
                    bump = (bump > 0.0f) ? bump * 10.0f : kEps;
                    continue;
                }
                info = lapack::SpotrsU(H, d, Af.data(), H, Bf.data(), H);
                if (info != 0) {
                    if (err)
                        *err = "UpdateCOneFromTrainLinkageListStreaming: potrs failed (info=" +
                            std::to_string(info) + ").";
                    return false;
                }
                W = std::move(Bf);
                ok = true;
                break;
            }
            if (!ok) {
                if (err) *err = "UpdateCOneFromTrainLinkageListStreaming: Cholesky failed for all bumps.";
                return false;
            }

            C_one_inout->d = d;
            C_one_inout->h_vec = h_vec_one;
            if (static_cast<int>(C_one_inout->books.size()) != m) {
                C_one_inout->books.resize(static_cast<std::size_t>(m));
            }
            for (int l = 0; l < m; ++l) {
                const int hl = std::max(1, h_vec_one[static_cast<std::size_t>(l)]);
                const int off = offsets_all[static_cast<std::size_t>(l)];
                ColMajorMatrix<float> book(d, hl);
                for (int code = 0; code < hl; ++code) {
                    const int p = off + code;
                    float* dst = book.Col(code);
                    for (int r = 0; r < d; ++r) {
                        dst[r] = W[static_cast<std::size_t>(p) + static_cast<std::size_t>(r) * static_cast<std::size_t>(
                            H)];
                    }
                }
                C_one_inout->books[static_cast<std::size_t>(l)] = std::move(book);
            }
            BumpCodebookBuildTag(C_one_inout);
            return true;
        }

        // Serial fallback (including GPU kernels, which are not thread-safe here).
        // Normal-equation accumulation for the exact joint least-squares solve.
        // This matches the in-memory behavior of UpdateCodebooksLS(R_in_sub, B_sub, a_sub, h_vec_one)
        // where R_in_sub are residuals (x - R_parent) for depth>0 nodes only.
        std::vector<double> Gd_all(static_cast<std::size_t>(H) * static_cast<std::size_t>(H), 0.0); // col-major
        std::vector<double> Td_all(static_cast<std::size_t>(H) * static_cast<std::size_t>(d), 0.0); // col-major

        auto Gd = [&](int r, int c) -> double& {
            return Gd_all[static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * static_cast<std::size_t>(H)];
        };
        auto Td = [&](int r, int c) -> double& {
            return Td_all[static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * static_cast<std::size_t>(H)];
        };

        std::vector<std::uint32_t> ids;
        std::vector<std::pair<std::uint32_t, int>> gid_pos;
        std::vector<std::uint32_t> real_ids;
        std::vector<std::uint32_t> parent;
        std::vector<std::uint32_t> depth_offsets;
        std::vector<std::uint8_t> codes_bytes;
        std::vector<float> coeffs_small;
        std::vector<std::uint8_t> code0_one_bytes;
        std::vector<float> a0;
        std::vector<std::uint8_t> virt_codes_bytes;
        std::vector<float> virt_coeffs;
        std::vector<float> virt_a0;

        ColMajorMatrix<std::uint8_t> x_u8;
        ColMajorMatrix<float> x_f32;
        ColMajorMatrix<float> Xrot;

        ColMajorMatrix<float> R_full; // d×n_real, depth-order
        ColMajorMatrix<float> R_virt; // d×n_virt, virtual roots reconstruction
        std::vector<float> tmp_self(static_cast<std::size_t>(d), 0.0f);
        std::vector<float> x_target(static_cast<std::size_t>(d), 0.0f);

        int idx_flat[32];
        float alpha[32];

        for (int cid = 0; cid < ivf.nlist(); ++cid) {
            const double w_cluster =
            (!train.is_bad_cluster.empty() &&
                cid >= 0 && cid < static_cast<int>(train.is_bad_cluster.size()) &&
                train.is_bad_cluster[static_cast<std::size_t>(cid)])
                ? static_cast<double>(cfg.virtual_cfg.alpha_bad)
                : 1.0;
            std::string local_err;
            if (!ivf_thr.ReadList(ivf, cid, &ids, &local_err)) {
                continue;
            }
            const int n_real = static_cast<int>(ids.size());
            if (n_real <= 0) {
                continue;
            }
            {
                std::string ignored;
                if (!linkage_thr.ReadCluster(cid, &real_ids, &parent, &depth_offsets,
                                             &codes_bytes, &coeffs_small,
                                             &code0_one_bytes, &a0,
                                             &virt_codes_bytes, &virt_coeffs, &virt_a0,
                                             &ignored)) {
                    continue;
                }
            }
            if (static_cast<int>(real_ids.size()) != n_real || static_cast<int>(parent.size()) != n_real) {
                continue;
            }
            if (depth_offsets.size() < 2) {
                continue;
            }
            const int depth1_start = static_cast<int>(depth_offsets[1]);
            if (depth1_start >= n_real) {
                continue;
            }

            // gid->pos map for list-order alignment.
            gid_pos.clear();
            gid_pos.reserve(static_cast<std::size_t>(n_real));
            for (int i = 0; i < n_real; ++i) {
                gid_pos.emplace_back(ids[static_cast<std::size_t>(i)], i);
            }
            std::sort(gid_pos.begin(), gid_pos.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });

            // Read raw vectors in list-order and rotate.
            if (is_f32_run) {
                if (has_raw_f32) {
                    if (!train_thr.ReadRawF32Span(ivf.Offset(cid), static_cast<std::uint32_t>(n_real), &x_f32,
                                                  &local_err)) {
                        LogWarn("UpdateCOne[cid=" + std::to_string(cid) + "]: raw f32 read failed: " + local_err);
                        continue;
                    }
                }
                else if (ReaderCanReadF32(fallback_reader)) {
                    if (!io::ReadDatasetVectorByIdsF32(*fallback_reader, ids, &x_f32, &local_err)) {
                        LogWarn(
                            "UpdateCOne[cid=" + std::to_string(cid) + "]: fvecs fallback read failed: " + local_err);
                        continue;
                    }
                }
                else if (ReaderCanReadF32(fallback_reader)) {
                    if (!io::ReadDatasetVectorByIdsF32(*fallback_reader, ids, &x_f32, &local_err)) {
                        LogWarn("UpdateCOne[cid=" + std::to_string(cid) + "]: fbin fallback read failed: " + local_err);
                        continue;
                    }
                }
                else {
                    LogWarn("UpdateCOne[cid=" + std::to_string(cid) + "]: no f32 vector source available.");
                    continue;
                }
                Xrot.rows = train.R.rows;
                Xrot.cols = x_f32.cols;
                Xrot.data.resize(static_cast<std::size_t>(Xrot.rows) * static_cast<std::size_t>(Xrot.cols));
                kernels->Gemm(false, false, 1.0f, train.R, x_f32, 0.0f, &Xrot);
            }
            else if (is_u8_run) {
                if (has_raw_u8) {
                    if (!train_thr.ReadRawU8Span(ivf.Offset(cid), static_cast<std::uint32_t>(n_real), &x_u8,
                                                 &local_err)) {
                        LogWarn("UpdateCOne[cid=" + std::to_string(cid) + "]: raw u8 read failed: " + local_err);
                        continue;
                    }
                }
                else if (ReaderCanReadU8(fallback_reader)) {
                    if (!io::ReadDatasetVectorByIdsU8(*fallback_reader, ids, &x_u8, &local_err)) {
                        LogWarn(
                            "UpdateCOne[cid=" + std::to_string(cid) + "]: bvecs fallback read failed: " + local_err);
                        continue;
                    }
                }
                else {
                    LogWarn("UpdateCOne[cid=" + std::to_string(cid) + "]: no u8 vector source available.");
                    continue;
                }
                kernels->ConvertU8ToF32AndRotate(x_u8, train.R, &Xrot);
            }
            else {
                LogWarn("UpdateCOne[cid=" + std::to_string(cid) + "]: unknown vector dtype (neither f32 nor u8 run).");
                continue;
            }

            // Reconstruct R_full in depth-order using old codebooks (for parent reconstruction only).
            R_full.rows = d;
            R_full.cols = n_real;
            R_full.data.resize(static_cast<std::size_t>(d) * static_cast<std::size_t>(n_real));
            const std::uint8_t* codes_u8 = codes_bytes.data();
            const std::uint8_t* code0_one_u8 = code0_one_bytes.data();

            const int n_virt = (m_codes > 0) ? static_cast<int>(virt_coeffs.size()) / m_codes : 0;
            R_virt.rows = d;
            R_virt.cols = n_virt;
            R_virt.data.resize(static_cast<std::size_t>(d) * static_cast<std::size_t>(std::max(0, n_virt)));
            const std::uint8_t* vcodes_u8 = virt_codes_bytes.data();
            for (int v = 0; v < n_virt; ++v) {
                const std::uint8_t* cu8 = vcodes_u8 + static_cast<std::size_t>(v) * m_codes;
                const float* a_small = virt_coeffs.data() + static_cast<std::size_t>(v) * m_codes;
                const float a0v = (v < static_cast<int>(virt_a0.size())) ? virt_a0[static_cast<std::size_t>(v)] : 0.0f;
                AccumulateReconDepth0FromSmallU8(train.C_root, cid, cu8, a_small, a0v, m, d, R_virt.Col(v));
            }

            for (int i = 0; i < n_real; ++i) {
                const std::uint8_t* cu8 = codes_u8 + static_cast<std::size_t>(i) * m_codes;
                const float* a_small = coeffs_small.data() + static_cast<std::size_t>(i) * m_codes;
                const float a0i = a0[static_cast<std::size_t>(i)];
                const std::uint32_t p1 = parent[static_cast<std::size_t>(i)];
                float* dst = R_full.Col(i);
                if (p1 == 0) {
                    AccumulateReconDepth0FromSmallU8(train.C_root, cid, cu8, a_small, a0i, m, d, dst);
                }
                else {
                    const int code0 = static_cast<int>(code0_one_u8[static_cast<std::size_t>(i)]);
                    AccumulateReconDepthPosFromSmallU8(C_one_old, code0, cu8, a_small, a0i, m, d, tmp_self.data());
                    const int p = static_cast<int>(p1 - 1); // local id (virtual-front)
                    if (p < n_virt) {
                        // p >= 0 is guaranteed: p = p1-1 and p1 >= 1.
                        const float* rp = R_virt.Col(p);
                        const float* ts = tmp_self.data();
#pragma omp simd
                        for (int r = 0; r < d; ++r) {
                            dst[r] = rp[r] + ts[r];
                        }
                    }
                    else {
                        const int pr = p - n_virt; // pr >= 0 guaranteed (p >= n_virt)
                        if (pr < n_real) {
                            const float* rp = R_full.Col(pr);
                            const float* ts = tmp_self.data();
#pragma omp simd
                            for (int r = 0; r < d; ++r) {
                                dst[r] = rp[r] + ts[r];
                            }
                        }
                        else {
                            LogWarn("UpdateCOne[cid=" + std::to_string(cid) + "]: parent pr=" +
                                std::to_string(pr) + " >= n_real=" + std::to_string(n_real) +
                                " at node i=" + std::to_string(i) + "; linkage data corrupt.");
                            std::memcpy(dst, tmp_self.data(), sizeof(float) * static_cast<std::size_t>(d));
                        }
                    }
                }
            }

            // Accumulate normal equations for depth>0 nodes.
            for (int i = depth1_start; i < n_real; ++i) {
                const int pos = FindListPos(gid_pos, real_ids[static_cast<std::size_t>(i)]);
                if (pos < 0 || pos >= n_real) {
                    LogWarn("UpdateCOne[cid=" + std::to_string(cid) + "]: gid not found in list-order map at node i=" +
                        std::to_string(i) + "; skipping.");
                    continue;
                }
                const int p = static_cast<int>(parent[static_cast<std::size_t>(i)] - 1);
                // p >= 0 guaranteed (parent[i] >= 1 for depth>0). (p-n_virt) >= 0 if p >= n_virt.
                const float* rp = (p < n_virt)
                                      ? R_virt.Col(p)
                                      : ((p - n_virt < n_real) ? R_full.Col(p - n_virt) : nullptr);
                if (!rp) {
                    LogWarn("UpdateCOne[cid=" + std::to_string(cid) + "]: parent p=" + std::to_string(p) +
                        " out of range (n_virt=" + std::to_string(n_virt) + " n_real=" + std::to_string(n_real) +
                        ") at node i=" + std::to_string(i) + "; skipping.");
                    continue;
                }
                const float* x = Xrot.Col(pos);
                for (int r = 0; r < d; ++r) {
                    x_target[static_cast<std::size_t>(r)] = x[r] - rp[r];
                }

                const int code0 = static_cast<int>(code0_one_u8[static_cast<std::size_t>(i)]);
                idx_flat[0] = offsets_all[0] + code0;
                alpha[0] = a0[static_cast<std::size_t>(i)];

                const std::uint8_t* cu8 = codes_u8 + static_cast<std::size_t>(i) * m_codes;
                const float* a_small = coeffs_small.data() + static_cast<std::size_t>(i) * m_codes;
                for (int l = 1; l < m; ++l) {
                    const int idx = l - 1;
                    const int code = static_cast<int>(cu8[static_cast<std::size_t>(idx)]);
                    idx_flat[l] = offsets_all[static_cast<std::size_t>(l)] + code;
                    alpha[l] = a_small[static_cast<std::size_t>(idx)];
                }

                for (int u = 0; u < m; ++u) {
                    const int pu = idx_flat[u];
                    const auto au = static_cast<double>(alpha[u]);
                    for (int r = 0; r < d; ++r) {
                        Td(pu, r) += w_cluster * au *
                            static_cast<double>(x_target[static_cast<std::size_t>(r)]);
                    }
                    for (int v = 0; v <= u; ++v) {
                        const int pv = idx_flat[v];
                        const auto av = static_cast<double>(alpha[v]);
                        const int rr = std::min(pu, pv);
                        const int cc = std::max(pu, pv);
                        Gd(rr, cc) += w_cluster * au * av;
                    }
                }
            }
        }

        // Convert to float for LAPACK (upper triangle only).
        std::vector<float> G(static_cast<std::size_t>(H) * static_cast<std::size_t>(H), 0.0f);
        std::vector<float> W(static_cast<std::size_t>(H) * static_cast<std::size_t>(d), 0.0f);
        for (int c = 0; c < H; ++c) {
            for (int r = 0; r <= c; ++r) {
                G[static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * static_cast<std::size_t>(H)] =
                    static_cast<float>(Gd(r, c));
            }
        }
        for (int r = 0; r < H; ++r) {
            for (int c = 0; c < d; ++c) {
                W[static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * static_cast<std::size_t>(H)] =
                    static_cast<float>(Td(r, c));
            }
        }

        float bump = 0.0f;
        bool ok = false;
        for (int attempt = 0; attempt < 8; ++attempt) {
            std::vector<float> Af = G;
            if (bump > 0.0f) {
                for (int i = 0; i < H; ++i) {
                    Af[static_cast<std::size_t>(i) + static_cast<std::size_t>(i) * static_cast<std::size_t>(H)] += bump;
                }
            }
            std::vector<float> Bf = W;
            int info = lapack::SpotrfU(H, Af.data(), H);
            if (info != 0) {
                bump = (bump > 0.0f) ? bump * 10.0f : kEps;
                continue;
            }
            info = lapack::SpotrsU(H, d, Af.data(), H, Bf.data(), H);
            if (info != 0) {
                if (err)
                    *err = "UpdateCOneFromTrainLinkageListStreaming: potrs failed (info=" + std::to_string(info) +
                        ").";
                return false;
            }
            W.swap(Bf);
            ok = true;
            break;
        }
        if (!ok) {
            if (err) *err = "UpdateCOneFromTrainLinkageListStreaming: Cholesky failed for all bumps.";
            return false;
        }

        // Unpack solution into codebooks.
        C_one_inout->d = d;
        C_one_inout->h_vec = h_vec_one;
        if (static_cast<int>(C_one_inout->books.size()) != m) {
            C_one_inout->books.resize(static_cast<std::size_t>(m));
        }
        for (int l = 0; l < m; ++l) {
            const int hl = std::max(1, h_vec_one[static_cast<std::size_t>(l)]);
            const int off = offsets_all[static_cast<std::size_t>(l)];
            ColMajorMatrix<float> book(d, hl);
            for (int code = 0; code < hl; ++code) {
                const int p = off + code;
                float* dst = book.Col(code);
                for (int r = 0; r < d; ++r) {
                    dst[r] = W[static_cast<std::size_t>(p) + static_cast<std::size_t>(r) * static_cast<std::size_t>(H)];
                }
            }
            C_one_inout->books[static_cast<std::size_t>(l)] = std::move(book);
        }
        BumpCodebookBuildTag(C_one_inout);
        return true;
    }

    bool UpdateCOneFromInitLinkageListStreamingExactLS(const Config& cfg,
                                                       const TrainResult& train,
                                                       const io::BaseListReader& train_list,
                                                       const io::IvfListsReader& ivf,
                                                       const io::LinkageListReader& linkage_list,
                                                       const io::DatasetVectorReader* fallback_reader,
                                                       bool allow_random_fallback,
                                                       StreamKernelProvider* kernels,
                                                       CodebookPack* C_one_out,
                                                       std::string* err) {
        if (!kernels) {
            if (err) *err = "UpdateCOneFromInitLinkageListStreamingExactLS: kernels is null.";
            return false;
        }
        if (!C_one_out) {
            if (err) *err = "UpdateCOneFromInitLinkageListStreamingExactLS: C_one output is null.";
            return false;
        }
        const int d = train_list.meta().d;
        const int m = cfg.model.m;
        const int m_codes = std::max(0, m - 1);
        if (d <= 0 || m <= 1 || m_codes != linkage_list.m_codes()) {
            if (err) *err = "UpdateCOneFromInitLinkageListStreamingExactLS: invalid d/m/m_codes.";
            return false;
        }
        if (train.C_root.books.empty()) {
            if (err) *err = "UpdateCOneFromInitLinkageListStreamingExactLS: empty C_root.";
            return false;
        }

        const int h0_one = std::max(1, cfg.model.h0_one);
        const int h0_root = train.C_root.books.front().cols;
        if (h0_root <= 0) {
            if (err) *err = "UpdateCOneFromInitLinkageListStreamingExactLS: invalid h0_root.";
            return false;
        }
        if (h0_one > h0_root) {
            if (err) *err = "UpdateCOneFromInitLinkageListStreamingExactLS: h0_one must be <= h0_root.";
            return false;
        }

        // Geometry-aware root_id -> one_id mapping (only needed when sizes differ).
        //
        // IMPORTANT: when h0_one == h0_root, other pipelines treat layer0 codes as identical
        // and do NOT apply any mapping. Keep init-stage consistent to avoid training/encode mismatch.
        std::vector<std::uint16_t> root_to_one;
        if (h0_one != h0_root) {
            root_to_one.assign(static_cast<std::size_t>(h0_root), 0);
            ScopedOmpThreads omp_scope(1); // keep mapping stable across OMP settings
            KmeansConfig kcfg;
            kcfg.max_iters = 30;
            kcfg.tol = 1e-6f;
            kcfg.init_method = "kmeans++";
            kcfg.init_samples = std::max(1, h0_root);
            // Disable adaptive weights for stable mapping.
            kcfg.initial_weight = 1.0f;
            kcfg.min_weight = 1.0f;
            kcfg.outlier_quantile = 1.0f;
            kcfg.cost_threshold = 1e9f;
            kcfg.annealing_factor = 0.0f;
            kcfg.warmup_iters = 0;
            std::mt19937 rng(static_cast<std::mt19937::result_type>(cfg.train.seed ^ 0xC001D00Du));
            const KmeansResult km = SphericalKmeans(train.C_root.books.front(), h0_one, kcfg, &rng,
                                                    /*stream_kernels=*/nullptr,
                                                    /*profile_timing=*/false,
                                                    /*timing=*/nullptr);
            if (static_cast<int>(km.assignments.size()) != h0_root) {
                if (err)
                    *err =
                        "UpdateCOneFromInitLinkageListStreamingExactLS: unexpected root kmeans assignment size.";
                return false;
            }
            for (int r = 0; r < h0_root; ++r) {
                root_to_one[static_cast<std::size_t>(r)] =
                    static_cast<std::uint16_t>(km.assignments[static_cast<std::size_t>(r)]);
            }
        }

        io::LinkageListThreadReader linkage_thr;
        if (!linkage_thr.OpenFrom(linkage_list, err)) {
            return false;
        }
        io::BaseListThreadReader train_thr;
        if (!train_thr.OpenFrom(train_list, err)) {
            return false;
        }
        io::IvfListsThreadReader ivf_thr;
        if (!ivf_thr.OpenFrom(ivf, err)) {
            return false;
        }

        const bool has_raw_u8 = train_list.HasRawU8();
        const bool has_raw_f32 = train_list.HasRawF32();
        int code0_width_bytes = 0;
        if (!ValidateLinkageListCodeLayout(linkage_list,
                                           /*allow_wide_code0=*/true,
                                           "UpdateCOneFromInitLinkageListStreamingExactLS",
                                           &code0_width_bytes,
                                           err)) {
            return false;
        }

        auto h_vec_one = cfg.model.h_vec;
        if (!h_vec_one.empty()) {
            h_vec_one[0] = cfg.model.h0_one;
        }

        // Flatten offsets for all layers including l=0.
        std::vector<int> offsets_all(static_cast<std::size_t>(m), 0);
        int H = 0;
        for (int l = 0; l < m; ++l) {
            offsets_all[static_cast<std::size_t>(l)] = H;
            H += std::max(1, h_vec_one[static_cast<std::size_t>(l)]);
        }
        if (H <= 0) {
            if (err) *err = "UpdateCOneFromInitLinkageListStreamingExactLS: invalid flattened H.";
            return false;
        }

        auto PackUpper = [](int i, int j) -> std::size_t
        {
            // i <= j
            return static_cast<std::size_t>(j) * static_cast<std::size_t>(j + 1) / 2u + static_cast<std::size_t>(i);
        };
        const std::size_t packed_size = static_cast<std::size_t>(H) * static_cast<std::size_t>(H + 1) / 2u;
        const std::size_t td_size_t = static_cast<std::size_t>(H) * static_cast<std::size_t>(d); // transposed

        const int omp_max = OmpMaxThreads();
        int shards = cfg.runtime.c_one_update_shards_init_linkage;
        if (shards <= 0) {
            shards = cfg.runtime.c_one_update_shards;
        }
        if (shards <= 0) {
            shards = std::min(omp_max, 16);
        }
        shards = std::clamp(shards, 1, omp_max);

        const bool can_parallel_accum = (shards > 1 && omp_max > 1);
        if (can_parallel_accum) {
            // Sharded normal-equation accumulation to bound memory (O(shards * H^2)).
            // Keep BLAS single-threaded inside the OpenMP region to avoid oversubscription.
            ScopedBlasThreads blas_scope(1);
            ScopedOmpThreads omp_scope(shards);

            const std::size_t G_packed_all_size = packed_size * static_cast<std::size_t>(shards);
            auto G_packed_all_u = std::unique_ptr<double[]>(new double[G_packed_all_size]);
            double* G_packed_all = G_packed_all_u.get();
            const std::size_t T_all_t_size = td_size_t * static_cast<std::size_t>(shards);
            auto T_all_t_u = std::unique_ptr<double[]>(new double[T_all_t_size]);
            double* T_all_t = T_all_t_u.get();

#pragma omp parallel default(none) shared(G_packed_all, T_all_t, linkage_list, train_list, ivf, cfg, train, fallback_reader, offsets_all, root_to_one) firstprivate(PackUpper, packed_size, td_size_t, d, m, m_codes, h0_root, h0_one, has_raw_u8, has_raw_f32, code0_width_bytes, allow_random_fallback)
            {
                const int tid = omp_get_thread_num();
                double* G_packed = G_packed_all + static_cast<std::size_t>(tid) * packed_size;
                double* T_t = T_all_t + static_cast<std::size_t>(tid) * td_size_t;
                std::fill(G_packed, G_packed + packed_size, 0.0);
                std::fill(T_t, T_t + td_size_t, 0.0);

                auto Gd = [&](int r, int c) -> double& {
                    return G_packed[PackUpper(r, c)];
                };
                auto Td = [&](int p, int r) -> double& {
                    return T_t[static_cast<std::size_t>(p) * static_cast<std::size_t>(d) + static_cast<std::size_t>(r)];
                };

                // Per-thread IO readers + CPU kernels (thread-safe).
                io::LinkageListThreadReader linkage_thr_tls;
                io::BaseListThreadReader train_thr_tls;
                io::IvfListsThreadReader ivf_thr_tls;
                std::string open_err;
                if (!linkage_thr_tls.OpenFrom(linkage_list, &open_err) ||
                    !train_thr_tls.OpenFrom(train_list, &open_err) ||
                    !ivf_thr_tls.OpenFrom(ivf, &open_err)) {
                    // Leave local slices zero; reduction will still work and the solve will likely fail,
                    // but we avoid throwing from inside OMP.
                }

                CpuStreamKernels kernels_tls;
                StreamKernelProvider* kernels_local = &kernels_tls;

                std::vector<std::uint32_t> ids;
                std::vector<std::pair<std::uint32_t, int>> gid_pos;
                std::vector<std::uint32_t> real_ids;
                std::vector<std::uint32_t> parent;
                std::vector<std::uint32_t> depth_offsets;
                std::vector<std::uint8_t> codes_bytes;
                std::vector<float> coeffs_small;
                std::vector<std::uint8_t> code0_one_bytes;
                std::vector<float> a0;
                std::vector<std::uint8_t> virt_codes_bytes;
                std::vector<float> virt_coeffs;
                std::vector<float> virt_a0;

                ColMajorMatrix<std::uint8_t> x_u8;
                ColMajorMatrix<float> x_f32;
                ColMajorMatrix<float> Xrot;
                ColMajorMatrix<float> R_full; // d×n_real, depth-order
                ColMajorMatrix<float> R_virt; // d×n_virt
                std::vector<float> tmp_self(static_cast<std::size_t>(d), 0.0f);
                std::vector<float> x_target(static_cast<std::size_t>(d), 0.0f);

                int idx_flat[32];
                float alpha[32];

                // Match the deterministic reduction contract used by the regular C_one update.
                // Dynamic scheduling changes which clusters contribute to each thread-local shard.
#pragma omp for schedule(static)
                for (int cid = 0; cid < ivf.nlist(); ++cid) {
                    const double w_cluster =
                    (!train.is_bad_cluster.empty() &&
                        cid >= 0 && cid < static_cast<int>(train.is_bad_cluster.size()) &&
                        train.is_bad_cluster[static_cast<std::size_t>(cid)])
                        ? static_cast<double>(cfg.virtual_cfg.alpha_bad)
                        : 1.0;

                    std::string local_err;
                    if (!ivf_thr_tls.ReadList(ivf, cid, &ids, &local_err)) {
                        continue;
                    }
                    const int n_real = static_cast<int>(ids.size());
                    if (n_real <= 0) {
                        continue;
                    }
                    {
                        std::string ignored;
                        if (!linkage_thr_tls.ReadCluster(cid, &real_ids, &parent, &depth_offsets,
                                                         &codes_bytes, &coeffs_small,
                                                         &code0_one_bytes, &a0,
                                                         &virt_codes_bytes, &virt_coeffs, &virt_a0,
                                                         &ignored)) {
                            continue;
                        }
                    }
                    if (static_cast<int>(real_ids.size()) != n_real || static_cast<int>(parent.size()) != n_real) {
                        continue;
                    }
                    if (depth_offsets.size() < 2) {
                        continue;
                    }
                    const int depth1_start = static_cast<int>(depth_offsets[1]);
                    if (depth1_start >= n_real) {
                        continue;
                    }

                    gid_pos.clear();
                    gid_pos.reserve(static_cast<std::size_t>(n_real));
                    for (int i = 0; i < n_real; ++i) {
                        gid_pos.emplace_back(ids[static_cast<std::size_t>(i)], i);
                    }
                    std::sort(gid_pos.begin(), gid_pos.end(),
                              [](const auto& a, const auto& b) { return a.first < b.first; });

                    if (has_raw_f32) {
                        if (!train_thr_tls.ReadRawF32Span(ivf.Offset(cid), static_cast<std::uint32_t>(n_real), &x_f32,
                                                          &local_err)) {
                            continue;
                        }
                        Xrot.rows = train.R.rows;
                        Xrot.cols = x_f32.cols;
                        Xrot.data.resize(static_cast<std::size_t>(Xrot.rows) * static_cast<std::size_t>(Xrot.cols));
                        kernels_local->Gemm(false, false, 1.0f, train.R, x_f32, 0.0f, &Xrot);
                    }
                    else if (has_raw_u8) {
                        if (!train_thr_tls.ReadRawU8Span(ivf.Offset(cid), static_cast<std::uint32_t>(n_real), &x_u8,
                                                         &local_err)) {
                            continue;
                        }
                        kernels_local->ConvertU8ToF32AndRotate(x_u8, train.R, &Xrot);
                    }
                    else if (ReaderCanReadF32(fallback_reader)) {
                        if (!io::ReadDatasetVectorByIdsF32(*fallback_reader, ids, &x_f32, &local_err)) {
                            continue;
                        }
                        Xrot.rows = train.R.rows;
                        Xrot.cols = x_f32.cols;
                        Xrot.data.resize(static_cast<std::size_t>(Xrot.rows) * static_cast<std::size_t>(Xrot.cols));
                        kernels_local->Gemm(false, false, 1.0f, train.R, x_f32, 0.0f, &Xrot);
                    }
                    else if (ReaderCanReadF32(fallback_reader)) {
                        if (!io::ReadDatasetVectorByIdsF32(*fallback_reader, ids, &x_f32, &local_err)) {
                            continue;
                        }
                        Xrot.rows = train.R.rows;
                        Xrot.cols = x_f32.cols;
                        Xrot.data.resize(static_cast<std::size_t>(Xrot.rows) * static_cast<std::size_t>(Xrot.cols));
                        kernels_local->Gemm(false, false, 1.0f, train.R, x_f32, 0.0f, &Xrot);
                    }
                    else {
                        if (!allow_random_fallback) {
                            continue;
                        }
                        if (ReaderCanReadU8(fallback_reader)) {
                            if (!io::ReadDatasetVectorByIdsU8(*fallback_reader, ids, &x_u8, &local_err)) {
                                continue;
                            }
                            kernels_local->ConvertU8ToF32AndRotate(x_u8, train.R, &Xrot);
                        }
                        else {
                            continue;
                        }
                    }

                    R_full.rows = d;
                    R_full.cols = n_real;
                    R_full.data.resize(static_cast<std::size_t>(d) * static_cast<std::size_t>(n_real));
                    const std::uint8_t* codes_u8 = codes_bytes.data();
                    const RootCodeView code0_view = RootCodeView::FromBytes(code0_one_bytes, code0_width_bytes);

                    const int n_virt = (m_codes > 0) ? static_cast<int>(virt_coeffs.size()) / m_codes : 0;
                    R_virt.rows = d;
                    R_virt.cols = n_virt;
                    R_virt.data.resize(static_cast<std::size_t>(d) * static_cast<std::size_t>(std::max(0, n_virt)));
                    if (!R_virt.data.empty()) {
                        std::fill(R_virt.data.begin(), R_virt.data.end(), 0.0f);
                    }

                    for (int v = 0; v < n_virt; ++v) {
                        const std::uint8_t* cu8 = virt_codes_bytes.data() + static_cast<std::size_t>(v) * m_codes;
                        const float* a_small = virt_coeffs.data() + static_cast<std::size_t>(v) * m_codes;
                        const float a0i = (v < static_cast<int>(virt_a0.size()))
                                              ? virt_a0[static_cast<std::size_t>(v)]
                                              : 0.0f;
                        AccumulateReconDepth0FromSmallU8(train.C_root, cid, cu8, a_small, a0i, m, d, R_virt.Col(v));
                    }

                    for (int i = 0; i < n_real; ++i) {
                        const std::uint8_t* cu8 = codes_u8 + static_cast<std::size_t>(i) * m_codes;
                        const float* a_small = coeffs_small.data() + static_cast<std::size_t>(i) * m_codes;
                        const float a0i = a0[static_cast<std::size_t>(i)];
                        const std::uint32_t p1 = parent[static_cast<std::size_t>(i)];
                        float* dst = R_full.Col(i);
                        if (p1 == 0) {
                            const int code0_root = code0_view.Get(static_cast<std::size_t>(i));
                            AccumulateReconDepth0FromSmallU8(train.C_root, code0_root, cu8, a_small, a0i, m, d, dst);
                        }
                        else {
                            const int code0_root = code0_view.Get(static_cast<std::size_t>(i));
                            AccumulateReconDepthPosFromSmallU8(train.C_root, code0_root, cu8, a_small, a0i, m, d,
                                                               tmp_self.data());
                            const int p = static_cast<int>(p1 - 1);
                            // p >= 0 guaranteed (p1 >= 1). (p-n_virt) >= 0 if p >= n_virt.
                            const float* rp = (p < n_virt)
                                                  ? R_virt.Col(p)
                                                  : ((p - n_virt < n_real) ? R_full.Col(p - n_virt) : nullptr);
                            if (rp) {
                                const float* ts = tmp_self.data();
#pragma omp simd
                                for (int r = 0; r < d; ++r) {
                                    dst[r] = rp[r] + ts[r];
                                }
                            }
                            else {
                                LogWarn("UpdateCOneInit[cid=" + std::to_string(cid) + "]: parent pr=" +
                                    std::to_string(p - n_virt) + " >= n_real=" + std::to_string(n_real) +
                                    " at node i=" + std::to_string(i) + "; linkage data corrupt.");
                                std::memcpy(dst, tmp_self.data(), sizeof(float) * static_cast<std::size_t>(d));
                            }
                        }
                    }

                    for (int i = depth1_start; i < n_real; ++i) {
                        const int pos = FindListPos(gid_pos, real_ids[static_cast<std::size_t>(i)]);
                        if (pos < 0 || pos >= n_real) {
                            continue;
                        }
                        const int p = static_cast<int>(parent[static_cast<std::size_t>(i)] - 1);
                        // p >= 0 guaranteed (parent[i] >= 1 for depth>0). (p-n_virt) >= 0 if p >= n_virt.
                        const float* rp = (p < n_virt)
                                              ? R_virt.Col(p)
                                              : ((p - n_virt < n_real) ? R_full.Col(p - n_virt) : nullptr);
                        if (!rp) {
                            LogWarn("UpdateCOneInit[cid=" + std::to_string(cid) + "]: parent p=" + std::to_string(p) +
                                " out of range (n_virt=" + std::to_string(n_virt) + " n_real=" + std::to_string(n_real)
                                +
                                ") at node i=" + std::to_string(i) + "; skipping.");
                            continue;
                        }
                        const float* x = Xrot.Col(pos);
                        for (int r = 0; r < d; ++r) {
                            x_target[static_cast<std::size_t>(r)] = x[r] - rp[r];
                        }

                        const int code0_root = code0_view.Get(static_cast<std::size_t>(i));
                        int code0_one = code0_root;
                        if (!root_to_one.empty()) {
                            if (code0_root < 0 || code0_root >= h0_root) {
                                LogWarn("UpdateCOneInit[cid=" + std::to_string(cid) + "]: code0_root=" +
                                    std::to_string(code0_root) + " out of range [0," + std::to_string(h0_root) +
                                    ") at node i=" + std::to_string(i) + "; skipping.");
                                continue;
                            }
                            code0_one = static_cast<int>(root_to_one[static_cast<std::size_t>(code0_root)]);
                        }
                        if (code0_one < 0 || code0_one >= h0_one) {
                            LogWarn("UpdateCOneInit[cid=" + std::to_string(cid) + "]: code0_one=" +
                                std::to_string(code0_one) + " out of range [0," + std::to_string(h0_one) +
                                ") at node i=" + std::to_string(i) + "; skipping.");
                            continue;
                        }
                        idx_flat[0] = offsets_all[0] + code0_one;
                        alpha[0] = a0[static_cast<std::size_t>(i)];

                        const std::uint8_t* cu8 = codes_u8 + static_cast<std::size_t>(i) * m_codes;
                        const float* a_small = coeffs_small.data() + static_cast<std::size_t>(i) * m_codes;
                        for (int l = 1; l < m; ++l) {
                            const int idx = l - 1;
                            const int code = static_cast<int>(cu8[static_cast<std::size_t>(idx)]);
                            idx_flat[l] = offsets_all[static_cast<std::size_t>(l)] + code;
                            alpha[l] = a_small[static_cast<std::size_t>(idx)];
                        }

                        for (int u = 0; u < m; ++u) {
                            const int pu = idx_flat[u];
                            const auto au = static_cast<double>(alpha[u]);
                            for (int r = 0; r < d; ++r) {
                                Td(pu, r) += w_cluster * au *
                                    static_cast<double>(x_target[static_cast<std::size_t>(r)]);
                            }
                            for (int v = 0; v <= u; ++v) {
                                const int pv = idx_flat[v];
                                const auto av = static_cast<double>(alpha[v]);
                                const int rr = std::min(pu, pv);
                                const int cc = std::max(pu, pv);
                                Gd(rr, cc) += w_cluster * au * av;
                            }
                        }
                    }
                }
            }

            // Reduce shards into slot 0.
#pragma omp parallel for default(none) shared(G_packed_all) firstprivate(packed_size, shards) schedule(static)
            for (std::int64_t i64 = 0; i64 < static_cast<std::int64_t>(packed_size); ++i64) {
                const auto i = static_cast<std::size_t>(i64);
                double sum = G_packed_all[i];
                for (int s = 1; s < shards; ++s) {
                    sum += G_packed_all[static_cast<std::size_t>(s) * packed_size + i];
                }
                G_packed_all[i] = sum;
            }
#pragma omp parallel for default(none) shared(T_all_t) firstprivate(td_size_t, shards) schedule(static)
            for (std::int64_t i64 = 0; i64 < static_cast<std::int64_t>(td_size_t); ++i64) {
                const auto i = static_cast<std::size_t>(i64);
                double sum = T_all_t[i];
                for (int s = 1; s < shards; ++s) {
                    sum += T_all_t[static_cast<std::size_t>(s) * td_size_t + i];
                }
                T_all_t[i] = sum;
            }

            std::vector<float> G(static_cast<std::size_t>(H) * static_cast<std::size_t>(H), 0.0f);
            std::vector<float> W(static_cast<std::size_t>(H) * static_cast<std::size_t>(d), 0.0f);
            for (int c = 0; c < H; ++c) {
                for (int r = 0; r <= c; ++r) {
                    G[static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * static_cast<std::size_t>(H)] =
                        static_cast<float>(G_packed_all[PackUpper(r, c)]);
                }
            }
            for (int p = 0; p < H; ++p) {
                const double* src = T_all_t + static_cast<std::size_t>(p) * static_cast<std::size_t>(d);
                for (int r = 0; r < d; ++r) {
                    W[static_cast<std::size_t>(p) + static_cast<std::size_t>(r) * static_cast<std::size_t>(H)] =
                        static_cast<float>(src[static_cast<std::size_t>(r)]);
                }
            }

            float bump = 0.0f;
            bool ok = false;
            for (int attempt = 0; attempt < 8; ++attempt) {
                std::vector<float> Af = G;
                if (bump > 0.0f) {
                    for (int i = 0; i < H; ++i) {
                        Af[static_cast<std::size_t>(i) + static_cast<std::size_t>(i) * static_cast<std::size_t>(H)] +=
                            bump;
                    }
                }
                std::vector<float> Bf = W;
                int info = lapack::SpotrfU(H, Af.data(), H);
                if (info != 0) {
                    bump = (bump > 0.0f) ? bump * 10.0f : kEps;
                    continue;
                }
                info = lapack::SpotrsU(H, d, Af.data(), H, Bf.data(), H);
                if (info != 0) {
                    if (err)
                        *err = "UpdateCOneFromInitLinkageListStreamingExactLS: potrs failed (info=" +
                            std::to_string(info) + ").";
                    return false;
                }
                W.swap(Bf);
                ok = true;
                break;
            }
            if (!ok) {
                if (err) *err = "UpdateCOneFromInitLinkageListStreamingExactLS: Cholesky failed for all bumps.";
                return false;
            }

            C_one_out->d = d;
            C_one_out->h_vec = h_vec_one;
            if (static_cast<int>(C_one_out->books.size()) != m) {
                C_one_out->books.resize(static_cast<std::size_t>(m));
            }
            for (int l = 0; l < m; ++l) {
                const int hl = std::max(1, h_vec_one[static_cast<std::size_t>(l)]);
                const int off = offsets_all[static_cast<std::size_t>(l)];
                ColMajorMatrix<float> book(d, hl);
                for (int code = 0; code < hl; ++code) {
                    const int p = off + code;
                    float* dst = book.Col(code);
                    for (int r = 0; r < d; ++r) {
                        dst[r] = W[static_cast<std::size_t>(p) + static_cast<std::size_t>(r) * static_cast<std::size_t>(
                            H)];
                    }
                }
                C_one_out->books[static_cast<std::size_t>(l)] = std::move(book);
            }
            BumpCodebookBuildTag(C_one_out);
            return true;
        }

        std::vector<double> Gd_all(static_cast<std::size_t>(H) * static_cast<std::size_t>(H), 0.0); // col-major
        std::vector<double> Td_all(static_cast<std::size_t>(H) * static_cast<std::size_t>(d), 0.0); // col-major
        auto Gd = [&](int r, int c) -> double& {
            return Gd_all[static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * static_cast<std::size_t>(H)];
        };
        auto Td = [&](int r, int c) -> double& {
            return Td_all[static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * static_cast<std::size_t>(H)];
        };

        std::vector<std::uint32_t> ids;
        std::vector<std::pair<std::uint32_t, int>> gid_pos;
        std::vector<std::uint32_t> real_ids;
        std::vector<std::uint32_t> parent;
        std::vector<std::uint32_t> depth_offsets;
        std::vector<std::uint8_t> codes_bytes;
        std::vector<float> coeffs_small;
        std::vector<std::uint8_t> code0_one_bytes;
        std::vector<float> a0;
        std::vector<std::uint8_t> virt_codes_bytes;
        std::vector<float> virt_coeffs;
        std::vector<float> virt_a0;

        ColMajorMatrix<std::uint8_t> x_u8;
        ColMajorMatrix<float> x_f32;
        ColMajorMatrix<float> Xrot;

        ColMajorMatrix<float> R_full; // d×n_real, depth-order
        ColMajorMatrix<float> R_virt; // d×n_virt
        std::vector<float> tmp_self(static_cast<std::size_t>(d), 0.0f);
        std::vector<float> x_target(static_cast<std::size_t>(d), 0.0f);

        int idx_flat[32];
        float alpha[32];

        for (int cid = 0; cid < ivf.nlist(); ++cid) {
            const double w_cluster =
            (!train.is_bad_cluster.empty() &&
                cid >= 0 && cid < static_cast<int>(train.is_bad_cluster.size()) &&
                train.is_bad_cluster[static_cast<std::size_t>(cid)])
                ? static_cast<double>(cfg.virtual_cfg.alpha_bad)
                : 1.0;

            std::string local_err;
            if (!ivf_thr.ReadList(ivf, cid, &ids, &local_err)) {
                continue;
            }
            const int n_real = static_cast<int>(ids.size());
            if (n_real <= 0) {
                continue;
            }
            {
                std::string ignored;
                if (!linkage_thr.ReadCluster(cid, &real_ids, &parent, &depth_offsets,
                                             &codes_bytes, &coeffs_small,
                                             &code0_one_bytes, &a0,
                                             &virt_codes_bytes, &virt_coeffs, &virt_a0,
                                             &ignored)) {
                    continue;
                }
            }
            if (static_cast<int>(real_ids.size()) != n_real || static_cast<int>(parent.size()) != n_real) {
                continue;
            }
            if (depth_offsets.size() < 2) {
                continue;
            }
            const int depth1_start = static_cast<int>(depth_offsets[1]);
            if (depth1_start >= n_real) {
                continue;
            }

            gid_pos.clear();
            gid_pos.reserve(static_cast<std::size_t>(n_real));
            for (int i = 0; i < n_real; ++i) {
                gid_pos.emplace_back(ids[static_cast<std::size_t>(i)], i);
            }
            std::sort(gid_pos.begin(), gid_pos.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });

            if (has_raw_f32) {
                if (!train_thr.ReadRawF32Span(ivf.Offset(cid), static_cast<std::uint32_t>(n_real), &x_f32,
                                              &local_err)) {
                    continue;
                }
                Xrot.rows = train.R.rows;
                Xrot.cols = x_f32.cols;
                Xrot.data.resize(static_cast<std::size_t>(Xrot.rows) * static_cast<std::size_t>(Xrot.cols));
                kernels->Gemm(false, false, 1.0f, train.R, x_f32, 0.0f, &Xrot);
            }
            else if (has_raw_u8) {
                if (!train_thr.ReadRawU8Span(ivf.Offset(cid), static_cast<std::uint32_t>(n_real), &x_u8, &local_err)) {
                    continue;
                }
                kernels->ConvertU8ToF32AndRotate(x_u8, train.R, &Xrot);
            }
            else if (ReaderCanReadF32(fallback_reader)) {
                if (!io::ReadDatasetVectorByIdsF32(*fallback_reader, ids, &x_f32, &local_err)) {
                    continue;
                }
                Xrot.rows = train.R.rows;
                Xrot.cols = x_f32.cols;
                Xrot.data.resize(static_cast<std::size_t>(Xrot.rows) * static_cast<std::size_t>(Xrot.cols));
                kernels->Gemm(false, false, 1.0f, train.R, x_f32, 0.0f, &Xrot);
            }
            else if (ReaderCanReadF32(fallback_reader)) {
                if (!io::ReadDatasetVectorByIdsF32(*fallback_reader, ids, &x_f32, &local_err)) {
                    continue;
                }
                Xrot.rows = train.R.rows;
                Xrot.cols = x_f32.cols;
                Xrot.data.resize(static_cast<std::size_t>(Xrot.rows) * static_cast<std::size_t>(Xrot.cols));
                kernels->Gemm(false, false, 1.0f, train.R, x_f32, 0.0f, &Xrot);
            }
            else {
                if (!allow_random_fallback) {
                    continue;
                }
                if (ReaderCanReadU8(fallback_reader)) {
                    if (!io::ReadDatasetVectorByIdsU8(*fallback_reader, ids, &x_u8, &local_err)) {
                        continue;
                    }
                    kernels->ConvertU8ToF32AndRotate(x_u8, train.R, &Xrot);
                }
                else {
                    continue;
                }
            }

            R_full.rows = d;
            R_full.cols = n_real;
            R_full.data.resize(static_cast<std::size_t>(d) * static_cast<std::size_t>(n_real));
            const std::uint8_t* codes_u8 = codes_bytes.data();
            const RootCodeView code0_view = RootCodeView::FromBytes(code0_one_bytes, code0_width_bytes);

            const int n_virt = (m_codes > 0) ? static_cast<int>(virt_coeffs.size()) / m_codes : 0;
            R_virt.rows = d;
            R_virt.cols = n_virt;
            R_virt.data.resize(static_cast<std::size_t>(d) * static_cast<std::size_t>(std::max(0, n_virt)));
            if (!R_virt.data.empty()) {
                std::fill(R_virt.data.begin(), R_virt.data.end(), 0.0f);
            }

            for (int v = 0; v < n_virt; ++v) {
                const std::uint8_t* cu8 = virt_codes_bytes.data() + static_cast<std::size_t>(v) * m_codes;
                const float* a_small = virt_coeffs.data() + static_cast<std::size_t>(v) * m_codes;
                const float a0i = (v < static_cast<int>(virt_a0.size())) ? virt_a0[static_cast<std::size_t>(v)] : 0.0f;
                AccumulateReconDepth0FromSmallU8(train.C_root, cid, cu8, a_small, a0i, m, d, R_virt.Col(v));
            }

            for (int i = 0; i < n_real; ++i) {
                const std::uint8_t* cu8 = codes_u8 + static_cast<std::size_t>(i) * m_codes;
                const float* a_small = coeffs_small.data() + static_cast<std::size_t>(i) * m_codes;
                const float a0i = a0[static_cast<std::size_t>(i)];
                const std::uint32_t p1 = parent[static_cast<std::size_t>(i)];
                float* dst = R_full.Col(i);
                if (p1 == 0) {
                    const int code0_root = code0_view.Get(static_cast<std::size_t>(i));
                    AccumulateReconDepth0FromSmallU8(train.C_root, code0_root, cu8, a_small, a0i, m, d, dst);
                }
                else {
                    const int code0_root = code0_view.Get(static_cast<std::size_t>(i));
                    AccumulateReconDepthPosFromSmallU8(train.C_root, code0_root, cu8, a_small, a0i, m, d,
                                                       tmp_self.data());
                    const int p = static_cast<int>(p1 - 1);
                    // p >= 0 guaranteed (p1 >= 1). (p-n_virt) >= 0 if p >= n_virt.
                    const float* rp = (p < n_virt)
                                          ? R_virt.Col(p)
                                          : ((p - n_virt < n_real) ? R_full.Col(p - n_virt) : nullptr);
                    if (rp) {
                        const float* ts = tmp_self.data();
#pragma omp simd
                        for (int r = 0; r < d; ++r) {
                            dst[r] = rp[r] + ts[r];
                        }
                    }
                    else {
                        LogWarn("UpdateCOneInit[cid=" + std::to_string(cid) + "]: parent pr=" +
                            std::to_string(p - n_virt) + " >= n_real=" + std::to_string(n_real) +
                            " at node i=" + std::to_string(i) + "; linkage data corrupt.");
                        std::memcpy(dst, tmp_self.data(), sizeof(float) * static_cast<std::size_t>(d));
                    }
                }
            }

            for (int i = depth1_start; i < n_real; ++i) {
                const int pos = FindListPos(gid_pos, real_ids[static_cast<std::size_t>(i)]);
                if (pos < 0 || pos >= n_real) {
                    continue;
                }
                const int p = static_cast<int>(parent[static_cast<std::size_t>(i)] - 1);
                // p >= 0 guaranteed (parent[i] >= 1 for depth>0). (p-n_virt) >= 0 if p >= n_virt.
                const float* rp = (p < n_virt)
                                      ? R_virt.Col(p)
                                      : ((p - n_virt < n_real) ? R_full.Col(p - n_virt) : nullptr);
                if (!rp) {
                    LogWarn("UpdateCOneInit[cid=" + std::to_string(cid) + "]: parent p=" + std::to_string(p) +
                        " out of range (n_virt=" + std::to_string(n_virt) + " n_real=" + std::to_string(n_real) +
                        ") at node i=" + std::to_string(i) + "; skipping.");
                    continue;
                }
                const float* x = Xrot.Col(pos);
                for (int r = 0; r < d; ++r) {
                    x_target[static_cast<std::size_t>(r)] = x[r] - rp[r];
                }

                const int code0_root = code0_view.Get(static_cast<std::size_t>(i));
                int code0_one = code0_root;
                if (!root_to_one.empty()) {
                    if (code0_root < 0 || code0_root >= h0_root) {
                        LogWarn("UpdateCOneInit[cid=" + std::to_string(cid) + "]: code0_root=" +
                            std::to_string(code0_root) + " out of range [0," + std::to_string(h0_root) +
                            ") at node i=" + std::to_string(i) + "; skipping.");
                        continue;
                    }
                    code0_one = static_cast<int>(root_to_one[static_cast<std::size_t>(code0_root)]);
                }
                if (code0_one < 0 || code0_one >= h0_one) {
                    LogWarn("UpdateCOneInit[cid=" + std::to_string(cid) + "]: code0_one=" +
                        std::to_string(code0_one) + " out of range [0," + std::to_string(h0_one) +
                        ") at node i=" + std::to_string(i) + "; skipping.");
                    continue;
                }
                idx_flat[0] = offsets_all[0] + code0_one;
                alpha[0] = a0[static_cast<std::size_t>(i)];

                const std::uint8_t* cu8 = codes_u8 + static_cast<std::size_t>(i) * m_codes;
                const float* a_small = coeffs_small.data() + static_cast<std::size_t>(i) * m_codes;
                for (int l = 1; l < m; ++l) {
                    const int idx = l - 1;
                    const int code = static_cast<int>(cu8[static_cast<std::size_t>(idx)]);
                    idx_flat[l] = offsets_all[static_cast<std::size_t>(l)] + code;
                    alpha[l] = a_small[static_cast<std::size_t>(idx)];
                }

                for (int u = 0; u < m; ++u) {
                    const int pu = idx_flat[u];
                    const auto au = static_cast<double>(alpha[u]);
                    for (int r = 0; r < d; ++r) {
                        Td(pu, r) += w_cluster * au *
                            static_cast<double>(x_target[static_cast<std::size_t>(r)]);
                    }
                    for (int v = 0; v <= u; ++v) {
                        const int pv = idx_flat[v];
                        const auto av = static_cast<double>(alpha[v]);
                        const int rr = std::min(pu, pv);
                        const int cc = std::max(pu, pv);
                        Gd(rr, cc) += w_cluster * au * av;
                    }
                }
            }
        }

        std::vector<float> G(static_cast<std::size_t>(H) * static_cast<std::size_t>(H), 0.0f);
        std::vector<float> W(static_cast<std::size_t>(H) * static_cast<std::size_t>(d), 0.0f);
        for (int c = 0; c < H; ++c) {
            for (int r = 0; r <= c; ++r) {
                G[static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * static_cast<std::size_t>(H)] =
                    static_cast<float>(Gd(r, c));
            }
        }
        for (int r = 0; r < H; ++r) {
            for (int c = 0; c < d; ++c) {
                W[static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * static_cast<std::size_t>(H)] =
                    static_cast<float>(Td(r, c));
            }
        }

        float bump = 0.0f;
        bool ok = false;
        for (int attempt = 0; attempt < 8; ++attempt) {
            std::vector<float> Af = G;
            if (bump > 0.0f) {
                for (int i = 0; i < H; ++i) {
                    Af[static_cast<std::size_t>(i) + static_cast<std::size_t>(i) * static_cast<std::size_t>(H)] += bump;
                }
            }
            std::vector<float> Bf = W;
            int info = lapack::SpotrfU(H, Af.data(), H);
            if (info != 0) {
                bump = (bump > 0.0f) ? bump * 10.0f : kEps;
                continue;
            }
            info = lapack::SpotrsU(H, d, Af.data(), H, Bf.data(), H);
            if (info != 0) {
                if (err)
                    *err = "UpdateCOneFromInitLinkageListStreamingExactLS: potrs failed (info=" +
                        std::to_string(info) + ").";
                return false;
            }
            W.swap(Bf);
            ok = true;
            break;
        }
        if (!ok) {
            if (err) *err = "UpdateCOneFromInitLinkageListStreamingExactLS: Cholesky failed for all bumps.";
            return false;
        }

        C_one_out->d = d;
        C_one_out->h_vec = h_vec_one;
        if (static_cast<int>(C_one_out->books.size()) != m) {
            C_one_out->books.resize(static_cast<std::size_t>(m));
        }
        for (int l = 0; l < m; ++l) {
            const int hl = std::max(1, h_vec_one[static_cast<std::size_t>(l)]);
            const int off = offsets_all[static_cast<std::size_t>(l)];
            ColMajorMatrix<float> book(d, hl);
            for (int code = 0; code < hl; ++code) {
                const int p = off + code;
                float* dst = book.Col(code);
                for (int r = 0; r < d; ++r) {
                    dst[r] = W[static_cast<std::size_t>(p) + static_cast<std::size_t>(r) * static_cast<std::size_t>(H)];
                }
            }
            C_one_out->books[static_cast<std::size_t>(l)] = std::move(book);
        }
        BumpCodebookBuildTag(C_one_out);
        return true;
    }

    bool UpdateOpqRotationStreamingByCluster(const Config& cfg,
                                             const TrainResult& train,
                                             const io::BaseListReader& train_list,
                                             const io::IvfListsReader& ivf,
                                             const io::LinkageListReader& linkage_list,
                                             const io::DatasetVectorReader* fallback_reader,
                                             bool allow_random_fallback,
                                             StreamKernelProvider* kernels,
                                             ColMajorMatrix<float>* R_inout,
                                             std::string* err) {
        if (!kernels) {
            if (err) *err = "UpdateOpqRotationStreamingByCluster: kernels is null.";
            return false;
        }
        if (!R_inout) {
            if (err) *err = "UpdateOpqRotationStreamingByCluster: R output is null.";
            return false;
        }
        const int d = train_list.meta().d;
        const int m = cfg.model.m;
        const int m_codes = std::max(0, m - 1);
        if (d <= 0 || m <= 1) {
            if (err) *err = "UpdateOpqRotationStreamingByCluster: invalid d/m.";
            return false;
        }

        // Fast CPU path: reuse the same cluster-parallel reconstruction machinery used by the
        // post-C_one analysis to accumulate OPQ cross-covariance (and optionally also metrics),
        // then solve R = U*V^T. This avoids thousands of tiny GEMMs in a single-thread loop.
        //
        // NOTE: The analysis path uses per-thread CPU kernels for thread safety, so this branch
        // is intended for CPU kernels (train uses CPU kernels for linkage stages).
        if (!kernels->IsGpu()) {
            ColMajorMatrix<float> M;
            if (!AnalyzeTrainLinkageListAfterCOneUpdateStreaming(cfg,
                                                                 train,
                                                                 train_list,
                                                                 ivf,
                                                                 linkage_list,
                                                                 /*code0_one_bytes_are_root_codes=*/false,
                                                                 fallback_reader,
                                                                 allow_random_fallback,
                                                                 kernels,
                                                                 /*depth_stats_out=*/nullptr,
                                                                 /*mse_stats_out=*/nullptr,
                                                                 /*opq_M_out=*/&M,
                                                                 err)) {
                return false;
            }
            return UpdateOpqRotationFromCrossCov(M, R_inout, err);
        }

        ColMajorMatrix<float> M(d, d);
        std::fill(M.data.begin(), M.data.end(), 0.0f);

        io::LinkageListThreadReader linkage_thr;
        if (!linkage_thr.OpenFrom(linkage_list, err)) {
            return false;
        }
        io::BaseListThreadReader train_thr;
        if (!train_thr.OpenFrom(train_list, err)) {
            return false;
        }
        io::IvfListsThreadReader ivf_thr;
        if (!ivf_thr.OpenFrom(ivf, err)) {
            return false;
        }

        std::vector<std::uint32_t> ids;
        std::vector<std::pair<std::uint32_t, int>> gid_pos;
        std::vector<std::uint32_t> real_ids;
        std::vector<std::uint32_t> parent;
        std::vector<std::uint32_t> depth_offsets;
        std::vector<std::uint8_t> codes_bytes;
        std::vector<float> coeffs_small;
        std::vector<std::uint8_t> code0_one_bytes;
        std::vector<float> a0;
        std::vector<std::uint8_t> virt_codes_bytes;
        std::vector<float> virt_coeffs;
        std::vector<float> virt_a0;

        ColMajorMatrix<std::uint8_t> x_u8;
        ColMajorMatrix<float> X; // unrotated float, list-order
        ColMajorMatrix<float> Z_list; // rotated recon, list-order
        ColMajorMatrix<float> R_full; // rotated recon, depth-order
        ColMajorMatrix<float> R_virt; // rotated recon, virt roots (depth==0)
        ColMajorMatrix<float> Mt(d, d); // temporary for Z*X^T accumulation
        Mt.data.resize(static_cast<std::size_t>(d) * static_cast<std::size_t>(d));
        std::vector<float> tmp_self(static_cast<std::size_t>(d), 0.0f);

        // Prefer sequential list-order reads when available (see UpdateCOneFromTrainLinkageListStreaming).
        const bool has_raw_u8 = train_list.HasRawU8();
        const bool has_raw_f32 = train_list.HasRawF32();
        int code0_width_bytes = 0;
        if (!ValidateLinkageListCodeLayout(linkage_list,
                                           /*allow_wide_code0=*/false,
                                           "UpdateOpqRotationStreamingByCluster",
                                           &code0_width_bytes,
                                           err)) {
            return false;
        }

        for (int cid = 0; cid < ivf.nlist(); ++cid) {
            std::string local_err;
            if (!ivf_thr.ReadList(ivf, cid, &ids, &local_err)) {
                continue;
            }
            const int n_real = static_cast<int>(ids.size());
            if (n_real <= 0) {
                continue;
            }
            {
                std::string ignored;
                if (!linkage_thr.ReadCluster(cid, &real_ids, &parent, &depth_offsets,
                                             &codes_bytes, &coeffs_small,
                                             &code0_one_bytes, &a0,
                                             &virt_codes_bytes, &virt_coeffs, &virt_a0,
                                             &ignored)) {
                    continue;
                }
            }
            if (static_cast<int>(real_ids.size()) != n_real || static_cast<int>(parent.size()) != n_real) {
                continue;
            }
            if (code0_one_bytes.size() !=
                static_cast<std::size_t>(n_real) * static_cast<std::size_t>(code0_width_bytes)) {
                if (err) {
                    *err = "UpdateOpqRotationStreamingByCluster: code0_one_bytes size mismatch for cluster " +
                        std::to_string(cid) + ".";
                }
                return false;
            }

            gid_pos.clear();
            gid_pos.reserve(static_cast<std::size_t>(n_real));
            for (int i = 0; i < n_real; ++i) {
                gid_pos.emplace_back(ids[static_cast<std::size_t>(i)], i);
            }
            std::sort(gid_pos.begin(), gid_pos.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });

            // Read raw (unrotated) vectors in list-order.
            if (has_raw_f32) {
                if (!train_thr.ReadRawF32Span(ivf.Offset(cid), static_cast<std::uint32_t>(n_real), &X, &local_err)) {
                    continue;
                }
            }
            else if (has_raw_u8) {
                if (!train_thr.ReadRawU8Span(ivf.Offset(cid), static_cast<std::uint32_t>(n_real), &x_u8, &local_err)) {
                    continue;
                }
                kernels->ConvertU8ToF32(x_u8, &X);
            }
            else if (ReaderCanReadF32(fallback_reader)) {
                if (!io::ReadDatasetVectorByIdsF32(*fallback_reader, ids, &X, &local_err)) {
                    continue;
                }
            }
            else if (ReaderCanReadF32(fallback_reader)) {
                if (!io::ReadDatasetVectorByIdsF32(*fallback_reader, ids, &X, &local_err)) {
                    continue;
                }
            }
            else {
                if (!allow_random_fallback) {
                    continue;
                }
                if (ReaderCanReadU8(fallback_reader)) {
                    if (!io::ReadDatasetVectorByIdsU8(*fallback_reader, ids, &x_u8, &local_err)) {
                        continue;
                    }
                    kernels->ConvertU8ToF32(x_u8, &X);
                }
                else {
                    continue;
                }
            }

            // Reconstruct linkage Zhat in rotated space (depth-order), then scatter to list-order.
            R_full.rows = d;
            R_full.cols = n_real;
            R_full.data.resize(static_cast<std::size_t>(d) * static_cast<std::size_t>(n_real));
            Z_list.rows = d;
            Z_list.cols = n_real;
            // Filled column-wise below; avoid a full memset/zero-fill.
            Z_list.data.resize(static_cast<std::size_t>(d) * static_cast<std::size_t>(n_real));

            const std::uint8_t* codes_u8 = codes_bytes.data();
            const std::uint8_t* code0_one_u8 = code0_one_bytes.data();

            const int n_virt = (m_codes > 0) ? static_cast<int>(virt_coeffs.size()) / m_codes : 0;
            R_virt.rows = d;
            R_virt.cols = n_virt;
            R_virt.data.resize(static_cast<std::size_t>(d) * static_cast<std::size_t>(std::max(0, n_virt)));
            const std::uint8_t* vcodes_u8 = virt_codes_bytes.data();
            for (int v = 0; v < n_virt; ++v) {
                const std::uint8_t* cu8 = vcodes_u8 + static_cast<std::size_t>(v) * m_codes;
                const float* a_small = virt_coeffs.data() + static_cast<std::size_t>(v) * m_codes;
                const float a0v = (v < static_cast<int>(virt_a0.size())) ? virt_a0[static_cast<std::size_t>(v)] : 0.0f;
                AccumulateReconDepth0FromSmallU8(train.C_root, cid, cu8, a_small, a0v, m, d, R_virt.Col(v));
            }

            for (int i = 0; i < n_real; ++i) {
                const std::uint8_t* cu8 = codes_u8 + static_cast<std::size_t>(i) * m_codes;
                const float* a_small = coeffs_small.data() + static_cast<std::size_t>(i) * m_codes;
                const float a0i = a0[static_cast<std::size_t>(i)];
                const std::uint32_t p1 = parent[static_cast<std::size_t>(i)];
                float* dst = R_full.Col(i);
                if (p1 == 0) {
                    AccumulateReconDepth0FromSmallU8(train.C_root, cid, cu8, a_small, a0i, m, d, dst);
                }
                else {
                    const int code0 = static_cast<int>(code0_one_u8[static_cast<std::size_t>(i)]);
                    AccumulateReconDepthPosFromSmallU8(train.C_one, code0, cu8, a_small, a0i, m, d, tmp_self.data());
                    const int p = static_cast<int>(p1 - 1); // local id (virtual-front)
                    // p >= 0 guaranteed (p1 >= 1). (p-n_virt) >= 0 if p >= n_virt.
                    if (p < n_virt) {
                        const float* rp = R_virt.Col(p);
                        for (int r = 0; r < d; ++r) {
                            dst[r] = rp[r] + tmp_self[static_cast<std::size_t>(r)];
                        }
                    }
                    else {
                        const int pr = p - n_virt;
                        if (pr < n_real) {
                            const float* rp = R_full.Col(pr);
                            for (int r = 0; r < d; ++r) {
                                dst[r] = rp[r] + tmp_self[static_cast<std::size_t>(r)];
                            }
                        }
                        else {
                            LogWarn("UpdateOpqRotation[cid=" + std::to_string(cid) + "]: parent pr=" +
                                std::to_string(pr) + " >= n_real=" + std::to_string(n_real) +
                                " at node i=" + std::to_string(i) + "; linkage data corrupt.");
                            std::memcpy(dst, tmp_self.data(), sizeof(float) * static_cast<std::size_t>(d));
                        }
                    }
                }
                const int pos = FindListPos(gid_pos, real_ids[static_cast<std::size_t>(i)]);
                if (pos >= 0 && pos < n_real) {
                    std::memcpy(Z_list.Col(pos), dst, sizeof(float) * static_cast<std::size_t>(d));
                }
            }

            // M += Z_list * X^T
            kernels->Gemm(false, true, 1.0f, Z_list, X, 0.0f, &Mt);
            for (int c = 0; c < d; ++c) {
                for (int r = 0; r < d; ++r) {
                    M(r, c) += Mt(r, c);
                }
            }
        }

        ColMajorMatrix<float> U(d, d);
        ColMajorMatrix<float> VT(d, d);
        std::vector<float> S(static_cast<std::size_t>(d), 0.0f);
        ColMajorMatrix<float> Mcopy = M;
        int info = 0;
        {
            ScopedBlasThreads blas_scope(1);
            info = lapack::Sgesvd('S', 'S', d, d,
                                  Mcopy.data.data(), d, S.data(),
                                  U.data.data(), d, VT.data.data(), d);
        }
        if (info != 0) {
            if (err) *err = "UpdateOpqRotationStreamingByCluster: gesvd failed (info=" + std::to_string(info) + ").";
            return false;
        }
        R_inout->rows = d;
        R_inout->cols = d;
        R_inout->data.assign(static_cast<std::size_t>(d) * static_cast<std::size_t>(d), 0.0f);
        for (int c = 0; c < d; ++c) {
            float* dst = R_inout->Col(c);
            for (int r = 0; r < d; ++r) {
                float acc = 0.0f;
                for (int k = 0; k < d; ++k) {
                    acc += U(r, k) * VT(k, c);
                }
                dst[r] = acc;
            }
        }
        return true;
    }

    bool UpdateOpqRotationFromCrossCov(const ColMajorMatrix<float>& M,
                                       ColMajorMatrix<float>* R_inout,
                                       std::string* err) {
        if (!R_inout) {
            if (err) *err = "UpdateOpqRotationFromCrossCov: R output is null.";
            return false;
        }
        if (M.rows <= 0 || M.cols <= 0 || M.rows != M.cols) {
            if (err) *err = "UpdateOpqRotationFromCrossCov: M must be square.";
            return false;
        }
        const int d = M.rows;

        ColMajorMatrix<float> U(d, d);
        ColMajorMatrix<float> VT(d, d);
        std::vector<float> S(static_cast<std::size_t>(d), 0.0f);
        ColMajorMatrix<float> Mcopy = M;
        int info = 0;
        {
            ScopedBlasThreads blas_scope(1);
            info = lapack::Sgesvd('S', 'S', d, d,
                                  Mcopy.data.data(), d, S.data(),
                                  U.data.data(), d, VT.data.data(), d);
        }
        if (info != 0) {
            if (err) *err = "UpdateOpqRotationFromCrossCov: gesvd failed (info=" + std::to_string(info) + ").";
            return false;
        }

        R_inout->rows = d;
        R_inout->cols = d;
        R_inout->data.assign(static_cast<std::size_t>(d) * static_cast<std::size_t>(d), 0.0f);
        // R = U * VT  (both d×d, column-major)
        for (int c = 0; c < d; ++c) {
            float* dst = R_inout->Col(c);
            for (int r = 0; r < d; ++r) {
                float acc = 0.0f;
                for (int k = 0; k < d; ++k) {
                    acc += U(r, k) * VT(k, c);
                }
                dst[r] = acc;
            }
        }
        return true;
    }

    bool AnalyzeTrainLinkageListAfterCOneUpdateStreaming(const Config& cfg,
                                                         const TrainResult& train,
                                                         const io::BaseListReader& train_list,
                                                         const io::IvfListsReader& ivf,
                                                         const io::LinkageListReader& linkage_list,
                                                         bool code0_one_bytes_are_root_codes,
                                                         const io::DatasetVectorReader* fallback_reader,
                                                         bool allow_random_fallback,
                                                         StreamKernelProvider* kernels,
                                                         LinkageDepthStats* depth_stats_out,
                                                         MseStats* mse_stats_out,
                                                         ColMajorMatrix<float>* opq_M_out,
                                                         std::string* err) {
        if (!kernels) {
            if (err) *err = "AnalyzeTrainLinkageListAfterCOneUpdateStreaming: kernels is null.";
            return false;
        }
        if (!depth_stats_out && !mse_stats_out && !opq_M_out) {
            return true;
        }
        const int d = train_list.meta().d;
        const int m = cfg.model.m;
        const int m_codes = std::max(0, m - 1);
        if (d <= 0 || m <= 1) {
            if (err) *err = "AnalyzeTrainLinkageListAfterCOneUpdateStreaming: invalid d/m.";
            return false;
        }
        if (m_codes != linkage_list.m_codes()) {
            if (err) *err = "AnalyzeTrainLinkageListAfterCOneUpdateStreaming: m_codes mismatch.";
            return false;
        }

        io::LinkageListThreadReader linkage_thr;
        if (!linkage_thr.OpenFrom(linkage_list, err)) {
            return false;
        }
        io::BaseListThreadReader train_thr;
        if (!train_thr.OpenFrom(train_list, err)) {
            return false;
        }
        io::IvfListsThreadReader ivf_thr;
        if (!ivf_thr.OpenFrom(ivf, err)) {
            return false;
        }

        const bool has_raw_u8 = train_list.HasRawU8();
        const bool has_raw_f32 = train_list.HasRawF32();
        int code0_width_bytes = 0;
        const int nlist = ivf.nlist();
        const bool want_opq_m = (opq_M_out != nullptr);
        if (!ValidateLinkageListCodeLayout(linkage_list,
                                           /*allow_wide_code0=*/code0_one_bytes_are_root_codes,
                                           "AnalyzeTrainLinkageListAfterCOneUpdateStreaming",
                                           &code0_width_bytes,
                                           err)) {
            return false;
        }

        struct AnalyzeCoverageStats
        {
            std::uint64_t clusters_total = 0;
            std::uint64_t clusters_ivf_read_fail = 0;
            std::uint64_t clusters_empty = 0;
            std::uint64_t clusters_linkage_read_fail = 0;
            std::uint64_t clusters_size_mismatch = 0;
            std::uint64_t clusters_depth_offsets_too_small = 0;
            std::uint64_t clusters_mse_requested = 0;
            std::uint64_t clusters_raw_read_fail = 0;
            std::uint64_t clusters_mse_done = 0;

            std::uint64_t vec_total_in_lists = 0; // sum n_real over clusters that we successfully read list for
            std::uint64_t vec_total_in_linkage = 0; // sum n_real over clusters that we successfully read linkage for
            std::uint64_t vec_mse_considered = 0; // n_real (looped) within clusters where MSE path reached
            std::uint64_t vec_mse_gid_miss = 0; // FindListPos miss
            std::uint64_t vec_mse_used = 0; // actually accumulated

            std::uint64_t clusters_mse_has_root_to_one_map = 0; // mapping applied (h0_one < h0_root)

            std::uint64_t clusters_opq_requested = 0;
            std::uint64_t clusters_opq_done = 0;
            std::uint64_t clusters_opq_gid_miss = 0;
            std::uint64_t clusters_opq_raw_read_fail = 0;

            void Add(const AnalyzeCoverageStats& o) {
                clusters_total += o.clusters_total;
                clusters_ivf_read_fail += o.clusters_ivf_read_fail;
                clusters_empty += o.clusters_empty;
                clusters_linkage_read_fail += o.clusters_linkage_read_fail;
                clusters_size_mismatch += o.clusters_size_mismatch;
                clusters_depth_offsets_too_small += o.clusters_depth_offsets_too_small;
                clusters_mse_requested += o.clusters_mse_requested;
                clusters_raw_read_fail += o.clusters_raw_read_fail;
                clusters_mse_done += o.clusters_mse_done;
                vec_total_in_lists += o.vec_total_in_lists;
                vec_total_in_linkage += o.vec_total_in_linkage;
                vec_mse_considered += o.vec_mse_considered;
                vec_mse_gid_miss += o.vec_mse_gid_miss;
                vec_mse_used += o.vec_mse_used;
                clusters_mse_has_root_to_one_map += o.clusters_mse_has_root_to_one_map;
                clusters_opq_requested += o.clusters_opq_requested;
                clusters_opq_done += o.clusters_opq_done;
                clusters_opq_gid_miss += o.clusters_opq_gid_miss;
                clusters_opq_raw_read_fail += o.clusters_opq_raw_read_fail;
            }
        };

        AnalyzeCoverageStats cov_total;
        const int reduce_threads = std::max(1, omp_get_max_threads());
        std::vector<AnalyzeCoverageStats> cov_by_thread(static_cast<std::size_t>(reduce_threads));
        std::vector<ColMajorMatrix<float>> opq_m_by_thread;
        if (want_opq_m) {
            opq_m_by_thread.resize(static_cast<std::size_t>(reduce_threads));
            for (int tid = 0; tid < reduce_threads; ++tid) {
                opq_m_by_thread[static_cast<std::size_t>(tid)].rows = d;
                opq_m_by_thread[static_cast<std::size_t>(tid)].cols = d;
                opq_m_by_thread[static_cast<std::size_t>(tid)].data.assign(
                    static_cast<std::size_t>(d) * static_cast<std::size_t>(d), 0.0f);
            }
        }

        // Avoid nested BLAS parallelism when we cluster-parallelize the analysis / OPQ accumulation.
        ScopedBlasThreads blas_scope((mse_stats_out || want_opq_m) ? 1 : std::max(1, omp_get_max_threads()));

        if (want_opq_m) {
            opq_M_out->rows = d;
            opq_M_out->cols = d;
            opq_M_out->data.assign(static_cast<std::size_t>(d) * static_cast<std::size_t>(d), 0.0f);
        }

        const int h0_root = (!train.C_root.books.empty()) ? train.C_root.books.front().cols : 0;
        const int h0_one = (!train.C_one.books.empty()) ? train.C_one.books.front().cols : 0;
        std::vector<std::uint16_t> root_to_one;
        if (mse_stats_out && code0_one_bytes_are_root_codes) {
            if (h0_one <= 0 || h0_root <= 0 || h0_one > h0_root) {
                if (err)
                    *err =
                        "AnalyzeTrainLinkageListAfterCOneUpdateStreaming: invalid h0_one/h0_root for root->one mapping.";
                return false;
            }
            if (h0_one < h0_root) {
                cov_total.clusters_mse_has_root_to_one_map = 1;
                root_to_one.assign(static_cast<std::size_t>(h0_root), 0);
                ScopedOmpThreads omp_scope(1); // keep mapping stable across OMP settings
                KmeansConfig kcfg;
                kcfg.max_iters = 30;
                kcfg.tol = 1e-6f;
                kcfg.init_method = "kmeans++";
                kcfg.init_samples = std::max(1, h0_root);
                // Disable adaptive weights for stable mapping.
                kcfg.initial_weight = 1.0f;
                kcfg.min_weight = 1.0f;
                kcfg.outlier_quantile = 1.0f;
                kcfg.cost_threshold = 1e9f;
                kcfg.annealing_factor = 0.0f;
                kcfg.warmup_iters = 0;
                std::mt19937 rng(static_cast<std::mt19937::result_type>(cfg.train.seed ^ 0xC001D00Du));
                const KmeansResult km = SphericalKmeans(train.C_root.books.front(), h0_one, kcfg, &rng,
                                                        /*stream_kernels=*/nullptr,
                                                        /*profile_timing=*/false,
                                                        /*timing=*/nullptr);
                if (static_cast<int>(km.assignments.size()) != h0_root) {
                    if (err)
                        *err =
                            "AnalyzeTrainLinkageListAfterCOneUpdateStreaming: failed to build root->one mapping.";
                    return false;
                }
                for (int r = 0; r < h0_root; ++r) {
                    root_to_one[static_cast<std::size_t>(r)] =
                        static_cast<std::uint16_t>(km.assignments[static_cast<std::size_t>(r)]);
                }
            }
        }

        std::vector<std::uint64_t> cluster_real(static_cast<std::size_t>(nlist), 0);
        std::vector<std::uint64_t> cluster_linkaged(static_cast<std::size_t>(nlist), 0);
        std::vector<double> cluster_depth_sum(static_cast<std::size_t>(nlist), 0.0);
        std::vector<int> cluster_max_depth(static_cast<std::size_t>(nlist), 0);
        std::vector<double> cluster_mse_sum;
        std::vector<double> cluster_mse_min;
        std::vector<double> cluster_mse_max;
        if (mse_stats_out) {
            cluster_mse_sum.assign(static_cast<std::size_t>(nlist), 0.0);
            cluster_mse_min.assign(static_cast<std::size_t>(nlist), std::numeric_limits<double>::infinity());
            cluster_mse_max.assign(static_cast<std::size_t>(nlist), -std::numeric_limits<double>::infinity());
        }

        std::atomic<bool> ok{true};
        std::string first_err;
        std::mutex err_mu;
        auto fail = [&](const std::string& msg)
        {
            ok.store(false, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lk(err_mu);
            if (first_err.empty()) first_err = msg;
        };

#pragma omp parallel default(none) shared(cov_by_thread, opq_m_by_thread, cluster_real, cluster_linkaged, cluster_depth_sum, cluster_max_depth, cluster_mse_sum, cluster_mse_min, cluster_mse_max, ok, first_err, err_mu, linkage_list, train_list, ivf, train, root_to_one, fallback_reader) firstprivate(fail, d, m, m_codes, nlist, has_raw_u8, has_raw_f32, code0_width_bytes, want_opq_m, mse_stats_out, code0_one_bytes_are_root_codes, allow_random_fallback)
        {
            const int tid = omp_get_thread_num();
            AnalyzeCoverageStats cov_local;
            ColMajorMatrix<float> M_local;
            if (want_opq_m) {
                M_local.rows = d;
                M_local.cols = d;
                M_local.data.assign(static_cast<std::size_t>(d) * static_cast<std::size_t>(d), 0.0f);
            }

            // IMPORTANT: `kernels` passed into this function may be a shared GPU stream-kernel provider
            // (e.g. `CudaStreamKernels`) which is not thread-safe. This analysis is cluster-parallel, so
            // use a per-thread CPU stream-kernel instance for rotation/conversion/GEMM to avoid races
            // (and resulting heap corruption) when omp_threads>1.
            CpuStreamKernels kernels_tls;
            StreamKernelProvider* kernels_local = &kernels_tls;

            io::LinkageListThreadReader linkage_thr;
            io::BaseListThreadReader train_thr;
            io::IvfListsThreadReader ivf_thr;
            std::string open_err;
            if (!linkage_thr.OpenFrom(linkage_list, &open_err)) {
                fail("AnalyzeTrainLinkageListAfterCOneUpdateStreaming: failed to open linkage thread reader: " +
                    open_err);
            }
            else if (!train_thr.OpenFrom(train_list, &open_err)) {
                fail("AnalyzeTrainLinkageListAfterCOneUpdateStreaming: failed to open train thread reader: " +
                    open_err);
            }
            else if (!ivf_thr.OpenFrom(ivf, &open_err)) {
                fail("AnalyzeTrainLinkageListAfterCOneUpdateStreaming: failed to open ivf thread reader: " + open_err);
            }

            std::vector<std::uint32_t> ids;
            std::vector<std::pair<std::uint32_t, int>> gid_pos;
            std::vector<std::uint32_t> real_ids;
            std::vector<std::uint32_t> parent;
            std::vector<std::uint32_t> depth_offsets;
            std::vector<std::uint8_t> codes_bytes;
            std::vector<float> coeffs_small;
            std::vector<std::uint8_t> code0_one_bytes;
            std::vector<float> a0;
            std::vector<std::uint8_t> virt_codes_bytes;
            std::vector<float> virt_coeffs;
            std::vector<float> virt_a0;

            ColMajorMatrix<std::uint8_t> x_u8;
            ColMajorMatrix<float> x_f32;
            ColMajorMatrix<float> X_u8;
            // unrotated float, list-order (for OPQ; only used for u8/ReaderCanReadU8(fallback_reader))
            const ColMajorMatrix<float>* X_ptr = nullptr;
            ColMajorMatrix<float> Xrot;
            ColMajorMatrix<float> R_full; // d×n_real, depth-order
            ColMajorMatrix<float> R_virt; // d×n_virt (virtual roots)
            ColMajorMatrix<float> Z_list; // d×n_real, list-order (for OPQ)
            std::vector<float> tmp_self(static_cast<std::size_t>(d), 0.0f);

            // Determinism note:
            // OPQ `M` is reduced later in a fixed serial order, so the cluster-to-thread assignment must also be fixed.
            // With dynamic scheduling, the grouping of clusters into each thread-local `M_local` changes across runs,
            // which changes floating-point accumulation order and can perturb `R` from iteration 2 onward.
#pragma omp for schedule(static)
            for (int cid = 0; cid < nlist; ++cid) {
                cov_local.clusters_total++;
                if (!ok.load(std::memory_order_relaxed)) {
                    continue;
                }
                std::string local_err;
                if (!ivf_thr.ReadList(ivf, cid, &ids, &local_err)) {
                    cov_local.clusters_ivf_read_fail++;
                    continue;
                }
                const int n_real = static_cast<int>(ids.size());
                if (n_real <= 0) {
                    cov_local.clusters_empty++;
                    continue;
                }
                cov_local.vec_total_in_lists += static_cast<std::uint64_t>(n_real);

                {
                    std::string ignored;
                    if (!linkage_thr.ReadCluster(cid, &real_ids, &parent, &depth_offsets,
                                                 &codes_bytes, &coeffs_small,
                                                 &code0_one_bytes, &a0,
                                                 &virt_codes_bytes, &virt_coeffs, &virt_a0,
                                                 &ignored)) {
                        cov_local.clusters_linkage_read_fail++;
                        continue;
                    }
                }
                if (static_cast<int>(real_ids.size()) != n_real || static_cast<int>(parent.size()) != n_real) {
                    cov_local.clusters_size_mismatch++;
                    continue;
                }
                if (depth_offsets.size() < 2) {
                    cov_local.clusters_depth_offsets_too_small++;
                    continue;
                }
                cov_local.vec_total_in_linkage += static_cast<std::uint64_t>(n_real);

                cluster_real[static_cast<std::size_t>(cid)] = static_cast<std::uint64_t>(n_real);
                const int depth0 = static_cast<int>(depth_offsets[1]);
                if (depth0 >= 0 && depth0 <= n_real) {
                    cluster_linkaged[static_cast<std::size_t>(cid)] = static_cast<std::uint64_t>(n_real - depth0);
                }

                const int depth_max_bound = static_cast<int>(depth_offsets.size()) - 2;
                int cluster_max_depth_local = 0;
                // depth_offsets may be padded up to max_depth; find the last non-empty depth bucket.
                for (int dep = depth_max_bound; dep >= 0; --dep) {
                    const std::uint32_t lo = depth_offsets[static_cast<std::size_t>(dep)];
                    const std::uint32_t hi = depth_offsets[static_cast<std::size_t>(dep + 1)];
                    if (hi > lo) {
                        cluster_max_depth_local = dep;
                        break;
                    }
                }
                cluster_max_depth[static_cast<std::size_t>(cid)] = cluster_max_depth_local;
                double depth_sum_local = 0.0;
                for (int dep = 0; dep <= depth_max_bound; ++dep) {
                    const std::uint32_t lo = depth_offsets[static_cast<std::size_t>(dep)];
                    const std::uint32_t hi = depth_offsets[static_cast<std::size_t>(dep + 1)];
                    if (hi > lo) {
                        depth_sum_local += static_cast<double>(dep) * static_cast<double>(hi - lo);
                    }
                }
                cluster_depth_sum[static_cast<std::size_t>(cid)] = depth_sum_local;

                if (!mse_stats_out && !want_opq_m) {
                    continue;
                }
                if (mse_stats_out) {
                    cov_local.clusters_mse_requested++;
                }
                if (want_opq_m) {
                    cov_local.clusters_opq_requested++;
                }

                // gid->pos map for list-order alignment.
                gid_pos.clear();
                gid_pos.reserve(static_cast<std::size_t>(n_real));
                for (int i = 0; i < n_real; ++i) {
                    gid_pos.emplace_back(ids[static_cast<std::size_t>(i)], i);
                }
                std::sort(gid_pos.begin(), gid_pos.end(),
                          [](const auto& a, const auto& b) { return a.first < b.first; });

                // Read raw vectors in list-order (unrotated) and rotate (for MSE).
                X_ptr = nullptr;
                if (has_raw_f32) {
                    if (!train_thr.ReadRawF32Span(ivf.Offset(cid), static_cast<std::uint32_t>(n_real), &x_f32,
                                                  &local_err)) {
                        cov_local.clusters_raw_read_fail++;
                        cov_local.clusters_opq_raw_read_fail += want_opq_m ? 1 : 0;
                        continue;
                    }
                    X_ptr = want_opq_m ? &x_f32 : nullptr;
                    if (mse_stats_out) {
                        Xrot.rows = train.R.rows;
                        Xrot.cols = x_f32.cols;
                        Xrot.data.resize(static_cast<std::size_t>(Xrot.rows) * static_cast<std::size_t>(Xrot.cols));
                        kernels_local->Gemm(false, false, 1.0f, train.R, x_f32, 0.0f, &Xrot);
                    }
                }
                else if (has_raw_u8) {
                    if (!train_thr.ReadRawU8Span(ivf.Offset(cid), static_cast<std::uint32_t>(n_real), &x_u8,
                                                 &local_err)) {
                        cov_local.clusters_raw_read_fail++;
                        cov_local.clusters_opq_raw_read_fail += want_opq_m ? 1 : 0;
                        continue;
                    }
                    X_ptr = nullptr;
                    if (want_opq_m) {
                        kernels_local->ConvertU8ToF32(x_u8, &X_u8);
                        X_ptr = &X_u8;
                    }
                    if (mse_stats_out) {
                        if (want_opq_m) {
                            Xrot.rows = train.R.rows;
                            Xrot.cols = X_ptr ? X_ptr->cols : 0;
                            Xrot.data.resize(static_cast<std::size_t>(Xrot.rows) * static_cast<std::size_t>(Xrot.cols));
                            kernels_local->Gemm(false, false, 1.0f, train.R, *X_ptr, 0.0f, &Xrot);
                        }
                        else {
                            kernels_local->ConvertU8ToF32AndRotate(x_u8, train.R, &Xrot);
                        }
                    }
                }
                else if (ReaderCanReadF32(fallback_reader)) {
                    if (!io::ReadDatasetVectorByIdsF32(*fallback_reader, ids, &x_f32, &local_err)) {
                        cov_local.clusters_raw_read_fail++;
                        cov_local.clusters_opq_raw_read_fail += want_opq_m ? 1 : 0;
                        continue;
                    }
                    X_ptr = want_opq_m ? &x_f32 : nullptr;
                    if (mse_stats_out) {
                        Xrot.rows = train.R.rows;
                        Xrot.cols = x_f32.cols;
                        Xrot.data.resize(static_cast<std::size_t>(Xrot.rows) * static_cast<std::size_t>(Xrot.cols));
                        kernels_local->Gemm(false, false, 1.0f, train.R, x_f32, 0.0f, &Xrot);
                    }
                }
                else if (ReaderCanReadF32(fallback_reader)) {
                    if (!io::ReadDatasetVectorByIdsF32(*fallback_reader, ids, &x_f32, &local_err)) {
                        cov_local.clusters_raw_read_fail++;
                        cov_local.clusters_opq_raw_read_fail += want_opq_m ? 1 : 0;
                        continue;
                    }
                    X_ptr = want_opq_m ? &x_f32 : nullptr;
                    if (mse_stats_out) {
                        Xrot.rows = train.R.rows;
                        Xrot.cols = x_f32.cols;
                        Xrot.data.resize(static_cast<std::size_t>(Xrot.rows) * static_cast<std::size_t>(Xrot.cols));
                        kernels_local->Gemm(false, false, 1.0f, train.R, x_f32, 0.0f, &Xrot);
                    }
                }
                else {
                    if (!allow_random_fallback) {
                        cov_local.clusters_raw_read_fail++;
                        cov_local.clusters_opq_raw_read_fail += want_opq_m ? 1 : 0;
                        continue;
                    }
                    if (ReaderCanReadU8(fallback_reader)) {
                        if (!io::ReadDatasetVectorByIdsU8(*fallback_reader, ids, &x_u8, &local_err)) {
                            cov_local.clusters_raw_read_fail++;
                            cov_local.clusters_opq_raw_read_fail += want_opq_m ? 1 : 0;
                            continue;
                        }
                        X_ptr = nullptr;
                        if (want_opq_m) {
                            kernels_local->ConvertU8ToF32(x_u8, &X_u8);
                            X_ptr = &X_u8;
                        }
                        if (mse_stats_out) {
                            if (want_opq_m) {
                                Xrot.rows = train.R.rows;
                                Xrot.cols = X_ptr ? X_ptr->cols : 0;
                                Xrot.data.resize(
                                    static_cast<std::size_t>(Xrot.rows) * static_cast<std::size_t>(Xrot.cols));
                                kernels_local->Gemm(false, false, 1.0f, train.R, *X_ptr, 0.0f, &Xrot);
                            }
                            else {
                                kernels_local->ConvertU8ToF32AndRotate(x_u8, train.R, &Xrot);
                            }
                        }
                    }
                    else {
                        cov_local.clusters_raw_read_fail++;
                        cov_local.clusters_opq_raw_read_fail += want_opq_m ? 1 : 0;
                        continue;
                    }
                }

                const std::uint8_t* codes_u8 = codes_bytes.data();
                const std::uint8_t* code0_one_u8 = code0_one_bytes.data();
                RootCodeView code0_view;
                if (code0_one_bytes_are_root_codes) {
                    code0_view = RootCodeView::FromBytes(code0_one_bytes, code0_width_bytes);
                }
                const auto read_code0 = [&](std::size_t idx) -> int
                {
                    if (code0_one_bytes_are_root_codes) {
                        return code0_view.Get(idx);
                    }
                    return static_cast<int>(code0_one_u8[idx]);
                };

                const int n_virt = (m_codes > 0) ? static_cast<int>(virt_coeffs.size()) / m_codes : 0;
                R_virt.rows = d;
                R_virt.cols = n_virt;
                R_virt.data.resize(static_cast<std::size_t>(d) * static_cast<std::size_t>(std::max(0, n_virt)));
                const std::uint8_t* vcodes_u8 = virt_codes_bytes.data();
                for (int v = 0; v < n_virt; ++v) {
                    const std::uint8_t* cu8 = vcodes_u8 + static_cast<std::size_t>(v) * m_codes;
                    const float* a_small = virt_coeffs.data() + static_cast<std::size_t>(v) * m_codes;
                    const float a0v = (v < static_cast<int>(virt_a0.size()))
                                          ? virt_a0[static_cast<std::size_t>(v)]
                                          : 0.0f;
                    AccumulateReconDepth0FromSmallU8(train.C_root, cid, cu8, a_small, a0v, m, d, R_virt.Col(v));
                }

                R_full.rows = d;
                R_full.cols = n_real;
                R_full.data.resize(static_cast<std::size_t>(d) * static_cast<std::size_t>(n_real));
                if (want_opq_m) {
                    Z_list.rows = d;
                    Z_list.cols = n_real;
                    Z_list.data.resize(static_cast<std::size_t>(d) * static_cast<std::size_t>(n_real));
                }

                double mse_sum_local = 0.0;
                double mse_min_local = std::numeric_limits<double>::infinity();
                double mse_max_local = -std::numeric_limits<double>::infinity();
                bool opq_gid_miss = false;

                for (int i = 0; i < n_real; ++i) {
                    const std::uint8_t* cu8 = codes_u8 + static_cast<std::size_t>(i) * m_codes;
                    const float* a_small = coeffs_small.data() + static_cast<std::size_t>(i) * m_codes;
                    const float a0i = a0[static_cast<std::size_t>(i)];
                    const std::uint32_t p1 = parent[static_cast<std::size_t>(i)];
                    float* dst = R_full.Col(i);
                    if (p1 == 0) {
                        if (code0_one_bytes_are_root_codes) {
                            const int code0_root = read_code0(static_cast<std::size_t>(i));
                            AccumulateReconDepth0FromSmallU8(train.C_root, code0_root, cu8, a_small, a0i, m, d, dst);
                        }
                        else {
                            AccumulateReconDepth0FromSmallU8(train.C_root, cid, cu8, a_small, a0i, m, d, dst);
                        }
                    }
                    else {
                        int code0 = read_code0(static_cast<std::size_t>(i));
                        if (code0_one_bytes_are_root_codes) {
                            if (!root_to_one.empty()) {
                                code0 = static_cast<int>(root_to_one[static_cast<std::size_t>(code0)]);
                            }
                        }
                        AccumulateReconDepthPosFromSmallU8(train.C_one, code0, cu8, a_small, a0i, m, d,
                                                           tmp_self.data());
                        const int p = static_cast<int>(p1 - 1); // local id (virtual-front)
                        // p >= 0 guaranteed (p1 >= 1). (p-n_virt) >= 0 if p >= n_virt.
                        if (p < n_virt) {
                            const float* rp = R_virt.Col(p);
                            for (int r = 0; r < d; ++r) {
                                dst[r] = rp[r] + tmp_self[static_cast<std::size_t>(r)];
                            }
                        }
                        else {
                            const int pr = p - n_virt;
                            if (pr < n_real) {
                                const float* rp = R_full.Col(pr);
                                for (int r = 0; r < d; ++r) {
                                    dst[r] = rp[r] + tmp_self[static_cast<std::size_t>(r)];
                                }
                            }
                            else {
                                LogWarn("AnalyzeLinkage[cid=" + std::to_string(cid) + "]: parent pr=" +
                                    std::to_string(pr) + " >= n_real=" + std::to_string(n_real) +
                                    " at node i=" + std::to_string(i) + "; linkage data corrupt.");
                                std::memcpy(dst, tmp_self.data(), sizeof(float) * static_cast<std::size_t>(d));
                            }
                        }
                    }

                    const int pos = FindListPos(gid_pos, real_ids[static_cast<std::size_t>(i)]);
                    if (pos < 0 || pos >= n_real) {
                        if (mse_stats_out) {
                            cov_local.vec_mse_gid_miss++;
                        }
                        if (want_opq_m) {
                            opq_gid_miss = true;
                        }
                        continue;
                    }
                    if (mse_stats_out) {
                        cov_local.vec_mse_used++;
                        const float* x = Xrot.Col(pos);
                        const float* z = dst;
                        double e = 0.0;
                        for (int r = 0; r < d; ++r) {
                            const double diff = static_cast<double>(x[r]) - static_cast<double>(z[r]);
                            e += diff * diff;
                        }
                        mse_sum_local += e;
                        mse_min_local = std::min(mse_min_local, e);
                        mse_max_local = std::max(mse_max_local, e);
                    }
                    if (want_opq_m) {
                        std::memcpy(Z_list.Col(pos), dst, sizeof(float) * static_cast<std::size_t>(d));
                    }
                }

                if (mse_stats_out) {
                    cov_local.vec_mse_considered += static_cast<std::uint64_t>(n_real);
                    cluster_mse_sum[static_cast<std::size_t>(cid)] = mse_sum_local;
                    cluster_mse_min[static_cast<std::size_t>(cid)] = mse_min_local;
                    cluster_mse_max[static_cast<std::size_t>(cid)] = mse_max_local;
                    cov_local.clusters_mse_done++;
                }
                if (want_opq_m) {
                    if (opq_gid_miss) {
                        cov_local.clusters_opq_gid_miss++;
                    }
                    else {
                        // M_local += Z_list * X^T
                        if (!X_ptr || X_ptr->rows != d || X_ptr->cols != n_real) {
                            cov_local.clusters_opq_raw_read_fail++;
                        }
                        else {
                            kernels_local->Gemm(false, true, 1.0f, Z_list, *X_ptr, 1.0f, &M_local);
                            cov_local.clusters_opq_done++;
                        }
                    }
                }
            }

            cov_by_thread[static_cast<std::size_t>(tid)] = cov_local;
            if (want_opq_m) {
                opq_m_by_thread[static_cast<std::size_t>(tid)] = std::move(M_local);
            }
        }

        for (int tid = 0; tid < reduce_threads; ++tid) {
            cov_total.Add(cov_by_thread[static_cast<std::size_t>(tid)]);
            if (want_opq_m) {
                const ColMajorMatrix<float>& M_local = opq_m_by_thread[static_cast<std::size_t>(tid)];
                for (int c = 0; c < d; ++c) {
                    for (int r = 0; r < d; ++r) {
                        (*opq_M_out)(r, c) += M_local(r, c);
                    }
                }
            }
        }

        if (!ok.load(std::memory_order_relaxed)) {
            if (err) {
                *err = first_err.empty() ? "AnalyzeTrainLinkageListAfterCOneUpdateStreaming: failed." : first_err;
            }
            return false;
        }

        std::uint64_t total_real = 0;
        std::uint64_t total_linkaged = 0;
        double depth_sum = 0.0;
        int max_depth = 0;
        double mse_sum = 0.0;
        double mse_min = std::numeric_limits<double>::infinity();
        double mse_max = -std::numeric_limits<double>::infinity();
        for (int cid = 0; cid < nlist; ++cid) {
            const auto i = static_cast<std::size_t>(cid);
            total_real += cluster_real[i];
            total_linkaged += cluster_linkaged[i];
            depth_sum += cluster_depth_sum[i];
            max_depth = std::max(max_depth, cluster_max_depth[i]);
            if (mse_stats_out) {
                mse_sum += cluster_mse_sum[i];
                mse_min = std::min(mse_min, cluster_mse_min[i]);
                mse_max = std::max(mse_max, cluster_mse_max[i]);
            }
        }

        if (depth_stats_out) {
            LinkageDepthStats stats;
            if (total_real > 0) {
                stats.linkage_ratio = static_cast<float>(static_cast<double>(total_linkaged) /
                    static_cast<double>(total_real));
                stats.max_depth = max_depth;
                stats.mean_depth = static_cast<float>(depth_sum / static_cast<double>(total_real));
            }
            *depth_stats_out = stats;
        }
        if (mse_stats_out) {
            MseStats stats;
            if (total_real > 0) {
                stats.mean = static_cast<float>(mse_sum / static_cast<double>(total_real));
                stats.min = static_cast<float>(std::isfinite(mse_min) ? mse_min : 0.0);
                stats.max = static_cast<float>(std::isfinite(mse_max) ? mse_max : 0.0);
            }
            *mse_stats_out = stats;
        }

        if (mse_stats_out) {
            const double vec_cov =
                (cov_total.vec_mse_considered > 0)
                    ? static_cast<double>(cov_total.vec_mse_used) / static_cast<double>(cov_total.vec_mse_considered)
                    : 0.0;
            LogInfo("Train linkage-summary MSE coverage: clusters_mse_done=" + std::to_string(
                    cov_total.clusters_mse_done) +
                "/" + std::to_string(cov_total.clusters_mse_requested) +
                " vec_used=" + std::to_string(cov_total.vec_mse_used) +
                "/" + std::to_string(cov_total.vec_mse_considered) +
                " (coverage=" + FormatDoubleLocal(100.0 * vec_cov, 2) + "%)" +
                " gid_miss=" + std::to_string(cov_total.vec_mse_gid_miss) +
                " root_to_one_map=" + std::to_string(cov_total.clusters_mse_has_root_to_one_map) +
                " skip{ivf_read_fail=" + std::to_string(cov_total.clusters_ivf_read_fail) +
                " empty=" + std::to_string(cov_total.clusters_empty) +
                " linkage_read_fail=" + std::to_string(cov_total.clusters_linkage_read_fail) +
                " size_mismatch=" + std::to_string(cov_total.clusters_size_mismatch) +
                " depth_offsets_small=" + std::to_string(cov_total.clusters_depth_offsets_too_small) +
                " raw_read_fail=" + std::to_string(cov_total.clusters_raw_read_fail) + "}");
        }

        // Signal to callers that OPQ accumulation did not succeed at all (e.g. missing raw data),
        // so they can fall back to the standalone OPQ update scan.
        if (want_opq_m && opq_M_out && cov_total.clusters_opq_done == 0) {
            opq_M_out->rows = 0;
            opq_M_out->cols = 0;
            opq_M_out->data.clear();
        }
        return true;
    }

    bool ComputeBadClusterMaskFromLinkageListStreaming(const Config& cfg,
                                                       const TrainResult& train,
                                                       const io::BaseListReader& train_list,
                                                       const io::IvfListsReader& ivf,
                                                       const io::LinkageListReader& linkage_list,
                                                       const io::DatasetVectorReader* fallback_reader,
                                                       bool allow_random_fallback,
                                                       StreamKernelProvider* kernels,
                                                       std::vector<bool>* is_bad_cluster_out,
                                                       double* baseline_out,
                                                       std::string* err) {
        if (!kernels) {
            if (err) *err = "ComputeBadClusterMaskFromLinkageListStreaming: kernels is null.";
            return false;
        }
        if (!is_bad_cluster_out) {
            if (err) *err = "ComputeBadClusterMaskFromLinkageListStreaming: is_bad_cluster_out is null.";
            return false;
        }
        if (baseline_out) {
            *baseline_out = 0.0;
        }

        const int d = train_list.meta().d;
        const int m = cfg.model.m;
        const int m_codes = std::max(0, m - 1);
        const int nlist = ivf.nlist();
        if (d <= 0 || m <= 1 || nlist <= 0) {
            if (err) *err = "ComputeBadClusterMaskFromLinkageListStreaming: invalid d/m/nlist.";
            return false;
        }
        if (m_codes != linkage_list.m_codes()) {
            if (err) *err = "ComputeBadClusterMaskFromLinkageListStreaming: m_codes mismatch.";
            return false;
        }

        io::LinkageListThreadReader linkage_thr;
        if (!linkage_thr.OpenFrom(linkage_list, err)) {
            return false;
        }
        io::BaseListThreadReader train_thr;
        if (!train_thr.OpenFrom(train_list, err)) {
            return false;
        }
        io::IvfListsThreadReader ivf_thr;
        if (!ivf_thr.OpenFrom(ivf, err)) {
            return false;
        }

        const bool has_raw_u8 = train_list.HasRawU8();
        const bool has_raw_f32 = train_list.HasRawF32();
        int code0_width_bytes = 0;
        if (!ValidateLinkageListCodeLayout(linkage_list,
                                           /*allow_wide_code0=*/false,
                                           "ComputeBadClusterMaskFromLinkageListStreaming",
                                           &code0_width_bytes,
                                           err)) {
            return false;
        }

        // Per-cluster MSE accumulators.
        std::vector<double> cluster_mse_sum(static_cast<std::size_t>(nlist), 0.0);
        std::vector<int> cluster_cnt(static_cast<std::size_t>(nlist), 0);

        // Thread-local buffers.
        std::vector<std::uint32_t> ids;
        std::vector<std::pair<std::uint32_t, int>> gid_pos;
        std::vector<std::uint32_t> real_ids;
        std::vector<std::uint32_t> parent;
        std::vector<std::uint32_t> depth_offsets;
        std::vector<std::uint8_t> codes_bytes;
        std::vector<float> coeffs_small;
        std::vector<std::uint8_t> code0_one_bytes;
        std::vector<float> a0;
        std::vector<std::uint8_t> virt_codes_bytes;
        std::vector<float> virt_coeffs;
        std::vector<float> virt_a0;

        ColMajorMatrix<std::uint8_t> x_u8;
        ColMajorMatrix<float> x_f32;
        ColMajorMatrix<float> Xrot;

        ColMajorMatrix<float> R_full;
        ColMajorMatrix<float> R_virt;
        std::vector<float> tmp_self(static_cast<std::size_t>(d), 0.0f);

        for (int cid = 0; cid < nlist; ++cid) {
            std::string local_err;
            if (!ivf_thr.ReadList(ivf, cid, &ids, &local_err)) {
                continue;
            }
            const int n_real = static_cast<int>(ids.size());
            if (n_real <= 0) {
                continue;
            }
            {
                std::string ignored;
                if (!linkage_thr.ReadCluster(cid, &real_ids, &parent, &depth_offsets,
                                             &codes_bytes, &coeffs_small,
                                             &code0_one_bytes, &a0,
                                             &virt_codes_bytes, &virt_coeffs, &virt_a0,
                                             &ignored)) {
                    continue;
                }
            }
            if (static_cast<int>(real_ids.size()) != n_real || static_cast<int>(parent.size()) != n_real) {
                continue;
            }
            if (depth_offsets.size() < 2) {
                continue;
            }

            // gid->pos map for list-order alignment.
            gid_pos.clear();
            gid_pos.reserve(static_cast<std::size_t>(n_real));
            for (int i = 0; i < n_real; ++i) {
                gid_pos.emplace_back(ids[static_cast<std::size_t>(i)], i);
            }
            std::sort(gid_pos.begin(), gid_pos.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });

            // Read raw vectors in list-order and rotate.
            if (has_raw_f32) {
                if (!train_thr.ReadRawF32Span(ivf.Offset(cid), static_cast<std::uint32_t>(n_real), &x_f32,
                                              &local_err)) {
                    continue;
                }
                Xrot.rows = train.R.rows;
                Xrot.cols = x_f32.cols;
                Xrot.data.resize(static_cast<std::size_t>(Xrot.rows) * static_cast<std::size_t>(Xrot.cols));
                kernels->Gemm(false, false, 1.0f, train.R, x_f32, 0.0f, &Xrot);
            }
            else if (has_raw_u8) {
                if (!train_thr.ReadRawU8Span(ivf.Offset(cid), static_cast<std::uint32_t>(n_real), &x_u8, &local_err)) {
                    continue;
                }
                kernels->ConvertU8ToF32AndRotate(x_u8, train.R, &Xrot);
            }
            else if (ReaderCanReadF32(fallback_reader)) {
                if (!io::ReadDatasetVectorByIdsF32(*fallback_reader, ids, &x_f32, &local_err)) {
                    continue;
                }
                Xrot.rows = train.R.rows;
                Xrot.cols = x_f32.cols;
                Xrot.data.resize(static_cast<std::size_t>(Xrot.rows) * static_cast<std::size_t>(Xrot.cols));
                kernels->Gemm(false, false, 1.0f, train.R, x_f32, 0.0f, &Xrot);
            }
            else if (ReaderCanReadF32(fallback_reader)) {
                if (!io::ReadDatasetVectorByIdsF32(*fallback_reader, ids, &x_f32, &local_err)) {
                    continue;
                }
                Xrot.rows = train.R.rows;
                Xrot.cols = x_f32.cols;
                Xrot.data.resize(static_cast<std::size_t>(Xrot.rows) * static_cast<std::size_t>(Xrot.cols));
                kernels->Gemm(false, false, 1.0f, train.R, x_f32, 0.0f, &Xrot);
            }
            else {
                if (!allow_random_fallback) {
                    continue;
                }
                if (ReaderCanReadU8(fallback_reader)) {
                    if (!io::ReadDatasetVectorByIdsU8(*fallback_reader, ids, &x_u8, &local_err)) {
                        continue;
                    }
                    kernels->ConvertU8ToF32AndRotate(x_u8, train.R, &Xrot);
                }
                else {
                    continue;
                }
            }

            const std::uint8_t* codes_u8 = codes_bytes.data();
            const std::uint8_t* code0_one_u8 = code0_one_bytes.data();

            const int n_virt = (m_codes > 0) ? static_cast<int>(virt_coeffs.size()) / m_codes : 0;
            R_virt.rows = d;
            R_virt.cols = n_virt;
            R_virt.data.resize(static_cast<std::size_t>(d) * static_cast<std::size_t>(std::max(0, n_virt)));
            const std::uint8_t* vcodes_u8 = virt_codes_bytes.data();
            for (int v = 0; v < n_virt; ++v) {
                const std::uint8_t* cu8 = vcodes_u8 + static_cast<std::size_t>(v) * m_codes;
                const float* a_small = virt_coeffs.data() + static_cast<std::size_t>(v) * m_codes;
                const float a0v = (v < static_cast<int>(virt_a0.size())) ? virt_a0[static_cast<std::size_t>(v)] : 0.0f;
                AccumulateReconDepth0FromSmallU8(train.C_root, cid, cu8, a_small, a0v, m, d, R_virt.Col(v));
            }

            R_full.rows = d;
            R_full.cols = n_real;
            R_full.data.resize(static_cast<std::size_t>(d) * static_cast<std::size_t>(n_real));

            for (int i = 0; i < n_real; ++i) {
                const std::uint8_t* cu8 = codes_u8 + static_cast<std::size_t>(i) * m_codes;
                const float* a_small = coeffs_small.data() + static_cast<std::size_t>(i) * m_codes;
                const float a0i = a0[static_cast<std::size_t>(i)];
                const std::uint32_t p1 = parent[static_cast<std::size_t>(i)];
                float* dst = R_full.Col(i);
                if (p1 == 0) {
                    AccumulateReconDepth0FromSmallU8(train.C_root, cid, cu8, a_small, a0i, m, d, dst);
                }
                else {
                    const int code0 = static_cast<int>(code0_one_u8[static_cast<std::size_t>(i)]);
                    AccumulateReconDepthPosFromSmallU8(train.C_one, code0, cu8, a_small, a0i, m, d, tmp_self.data());
                    const int p = static_cast<int>(p1 - 1); // local id (virtual-front)
                    // p >= 0 guaranteed (p1 >= 1). (p-n_virt) >= 0 if p >= n_virt.
                    if (p < n_virt) {
                        const float* rp = R_virt.Col(p);
                        for (int r = 0; r < d; ++r) {
                            dst[r] = rp[r] + tmp_self[static_cast<std::size_t>(r)];
                        }
                    }
                    else {
                        const int pr = p - n_virt;
                        if (pr < n_real) {
                            const float* rp = R_full.Col(pr);
                            for (int r = 0; r < d; ++r) {
                                dst[r] = rp[r] + tmp_self[static_cast<std::size_t>(r)];
                            }
                        }
                        else {
                            LogWarn("ComputeBadMask[cid=" + std::to_string(cid) + "]: parent pr=" +
                                std::to_string(pr) + " >= n_real=" + std::to_string(n_real) +
                                " at node i=" + std::to_string(i) + "; linkage data corrupt.");
                            std::memcpy(dst, tmp_self.data(), sizeof(float) * static_cast<std::size_t>(d));
                        }
                    }
                }
            }

            // Compute per-sample MSE and accumulate for this cluster.
            double cluster_sum = 0.0;
            int valid_cnt = 0;
            for (int i = 0; i < n_real; ++i) {
                const int pos = FindListPos(gid_pos, real_ids[static_cast<std::size_t>(i)]);
                if (pos < 0 || pos >= n_real) {
                    LogWarn("ComputeBadMask[cid=" + std::to_string(cid) + "]: gid=" +
                        std::to_string(real_ids[static_cast<std::size_t>(i)]) +
                        " not found in Xrot at node i=" + std::to_string(i) + "; skipping.");
                    continue;
                }
                const float* x = Xrot.Col(pos);
                const float* z = R_full.Col(i);
                double e = 0.0;
                for (int r = 0; r < d; ++r) {
                    const double diff = static_cast<double>(x[r]) - static_cast<double>(z[r]);
                    e += diff * diff;
                }
                cluster_sum += e;
                ++valid_cnt;
            }
            cluster_mse_sum[static_cast<std::size_t>(cid)] = cluster_sum;
            cluster_cnt[static_cast<std::size_t>(cid)] = valid_cnt;
        }

        // Compute per-cluster mean MSE.
        std::vector<double> cluster_avgs;
        cluster_avgs.reserve(static_cast<std::size_t>(nlist));
        for (int c = 0; c < nlist; ++c) {
            if (cluster_cnt[static_cast<std::size_t>(c)] > 0) {
                cluster_avgs.push_back(cluster_mse_sum[static_cast<std::size_t>(c)] /
                    static_cast<double>(cluster_cnt[static_cast<std::size_t>(c)]));
            }
        }

        // Compute quantile baseline.
        double baseline = 0.0;
        if (!cluster_avgs.empty()) {
            std::vector<double> sorted_avgs = cluster_avgs;
            std::sort(sorted_avgs.begin(), sorted_avgs.end());
            const double q = std::max(0.0, std::min(1.0, cfg.virtual_cfg.good_fraction));
            if (q <= 0.0) {
                baseline = sorted_avgs.front();
            }
            else if (q >= 1.0) {
                baseline = sorted_avgs.back();
            }
            else {
                const int n_used = static_cast<int>(sorted_avgs.size());
                int pos = static_cast<int>(std::llround(q * static_cast<double>(n_used)));
                pos = std::max(1, std::min(pos, n_used));
                baseline = sorted_avgs[static_cast<std::size_t>(pos - 1)];
            }
        }
        if (baseline_out) {
            *baseline_out = baseline;
        }

        // Build bad cluster mask.
        is_bad_cluster_out->assign(static_cast<std::size_t>(nlist), false);
        int bad_cnt = 0;
        for (int c = 0; c < nlist; ++c) {
            if (cluster_cnt[static_cast<std::size_t>(c)] <= 0) {
                continue;
            }
            const double avg = cluster_mse_sum[static_cast<std::size_t>(c)] /
                static_cast<double>(cluster_cnt[static_cast<std::size_t>(c)]);
            if (avg > baseline) {
                (*is_bad_cluster_out)[static_cast<std::size_t>(c)] = true;
                ++bad_cnt;
            }
        }

        LogInfo("Virtual init (streaming): num bad clusters = " + std::to_string(bad_cnt) +
            " / " + std::to_string(nlist) +
            " (baseline=" + std::to_string(baseline) + ")");

        return true;
    }

    bool ComputeBadClusterMaskFromInitLinkageListStreaming(const Config& cfg,
                                                           const TrainResult& train,
                                                           const io::BaseListReader& train_list,
                                                           const io::IvfListsReader& ivf,
                                                           const io::LinkageListReader& linkage_list,
                                                           const io::DatasetVectorReader* fallback_reader,
                                                           bool allow_random_fallback,
                                                           StreamKernelProvider* kernels,
                                                           std::vector<bool>* is_bad_cluster_out,
                                                           double* baseline_out,
                                                           std::string* err) {
        if (!kernels) {
            if (err) *err = "ComputeBadClusterMaskFromInitLinkageListStreaming: kernels is null.";
            return false;
        }
        if (!is_bad_cluster_out) {
            if (err) *err = "ComputeBadClusterMaskFromInitLinkageListStreaming: is_bad_cluster_out is null.";
            return false;
        }
        if (baseline_out) {
            *baseline_out = 0.0;
        }

        const int d = train_list.meta().d;
        const int m = cfg.model.m;
        const int m_codes = std::max(0, m - 1);
        const int nlist = ivf.nlist();
        if (d <= 0 || m <= 1 || nlist <= 0) {
            if (err) *err = "ComputeBadClusterMaskFromInitLinkageListStreaming: invalid d/m/nlist.";
            return false;
        }
        if (m_codes != linkage_list.m_codes()) {
            if (err) *err = "ComputeBadClusterMaskFromInitLinkageListStreaming: m_codes mismatch.";
            return false;
        }

        io::LinkageListThreadReader linkage_thr;
        if (!linkage_thr.OpenFrom(linkage_list, err)) {
            return false;
        }
        io::BaseListThreadReader train_thr;
        if (!train_thr.OpenFrom(train_list, err)) {
            return false;
        }
        io::IvfListsThreadReader ivf_thr;
        if (!ivf_thr.OpenFrom(ivf, err)) {
            return false;
        }

        const bool has_raw_u8 = train_list.HasRawU8();
        const bool has_raw_f32 = train_list.HasRawF32();
        int code0_width_bytes = 0;
        if (!ValidateLinkageListCodeLayout(linkage_list,
                                           /*allow_wide_code0=*/true,
                                           "ComputeBadClusterMaskFromInitLinkageListStreaming",
                                           &code0_width_bytes,
                                           err)) {
            return false;
        }

        std::vector<double> cluster_mse_sum(static_cast<std::size_t>(nlist), 0.0);
        std::vector<int> cluster_cnt(static_cast<std::size_t>(nlist), 0);

        struct MseThreadStats
        {
            std::uint64_t total_cnt = 0;
            double sum = 0.0;
            double min = std::numeric_limits<double>::infinity();
            double max = -std::numeric_limits<double>::infinity();
        };

        const int omp_max = std::max(1, OmpMaxThreads());
        std::vector<MseThreadStats> thread_stats(static_cast<std::size_t>(omp_max));
        std::atomic<bool> ok{true};
        std::string first_open_err;
        std::mutex open_err_mu;
        std::mutex fallback_mu;

#pragma omp parallel num_threads(omp_max) default(none) shared(linkage_list, train_list, ivf, train, fallback_reader, fallback_mu, open_err_mu, ok, first_open_err, thread_stats, cluster_mse_sum, cluster_cnt) firstprivate(d, m, m_codes, nlist, has_raw_u8, has_raw_f32, code0_width_bytes, allow_random_fallback)
        {
            const int tid = omp_get_thread_num();
            MseThreadStats local_stats;

            io::LinkageListThreadReader linkage_thr_tls;
            io::BaseListThreadReader train_thr_tls;
            io::IvfListsThreadReader ivf_thr_tls;
            {
                std::string local_open_err;
                if (!linkage_thr_tls.OpenFrom(linkage_list, &local_open_err) ||
                    !train_thr_tls.OpenFrom(train_list, &local_open_err) ||
                    !ivf_thr_tls.OpenFrom(ivf, &local_open_err)) {
                    ok.store(false);
                    const std::lock_guard<std::mutex> lock(open_err_mu);
                    if (first_open_err.empty()) {
                        first_open_err = local_open_err.empty()
                                             ? "init-linkage bad-mask: failed to open thread readers."
                                             : local_open_err;
                    }
                }
            }

            CpuStreamKernels kernels_tls;
            StreamKernelProvider* kernels_local = &kernels_tls;

            std::vector<std::uint32_t> ids;
            std::vector<std::pair<std::uint32_t, int>> gid_pos;
            std::vector<std::uint32_t> real_ids;
            std::vector<std::uint32_t> parent;
            std::vector<std::uint32_t> depth_offsets;
            std::vector<std::uint8_t> codes_bytes;
            std::vector<float> coeffs_small;
            std::vector<std::uint8_t> code0_one_bytes;
            std::vector<float> a0;
            std::vector<std::uint8_t> virt_codes_bytes;
            std::vector<float> virt_coeffs;
            std::vector<float> virt_a0;

            ColMajorMatrix<std::uint8_t> x_u8;
            ColMajorMatrix<float> x_f32;
            ColMajorMatrix<float> Xrot;
            ColMajorMatrix<float> R_full;
            ColMajorMatrix<float> R_virt;
            std::vector<float> tmp_self(static_cast<std::size_t>(d), 0.0f);

#pragma omp for schedule(guided, 16)
            for (int cid = 0; cid < nlist; ++cid) {
                if (!ok.load()) {
                    continue;
                }

                std::string local_err;
                if (!ivf_thr_tls.ReadList(ivf, cid, &ids, &local_err)) {
                    continue;
                }
                const int n_real = static_cast<int>(ids.size());
                if (n_real <= 0) {
                    continue;
                }
                {
                    std::string ignored;
                    if (!linkage_thr_tls.ReadCluster(cid, &real_ids, &parent, &depth_offsets,
                                                     &codes_bytes, &coeffs_small,
                                                     &code0_one_bytes, &a0,
                                                     &virt_codes_bytes, &virt_coeffs, &virt_a0,
                                                     &ignored)) {
                        continue;
                    }
                }
                if (static_cast<int>(real_ids.size()) != n_real || static_cast<int>(parent.size()) != n_real) {
                    continue;
                }

                // gid->pos map for list-order alignment.
                gid_pos.clear();
                gid_pos.reserve(static_cast<std::size_t>(n_real));
                for (int i = 0; i < n_real; ++i) {
                    gid_pos.emplace_back(ids[static_cast<std::size_t>(i)], i);
                }
                std::sort(gid_pos.begin(), gid_pos.end(),
                          [](const auto& a, const auto& b) { return a.first < b.first; });

                // Read raw vectors in list-order and rotate.
                if (has_raw_f32) {
                    if (!train_thr_tls.ReadRawF32Span(ivf.Offset(cid), static_cast<std::uint32_t>(n_real), &x_f32,
                                                      &local_err)) {
                        continue;
                    }
                    Xrot.rows = train.R.rows;
                    Xrot.cols = x_f32.cols;
                    Xrot.data.resize(static_cast<std::size_t>(Xrot.rows) * static_cast<std::size_t>(Xrot.cols));
                    kernels_local->Gemm(false, false, 1.0f, train.R, x_f32, 0.0f, &Xrot);
                }
                else if (has_raw_u8) {
                    if (!train_thr_tls.ReadRawU8Span(ivf.Offset(cid), static_cast<std::uint32_t>(n_real), &x_u8,
                                                     &local_err)) {
                        continue;
                    }
                    kernels_local->ConvertU8ToF32AndRotate(x_u8, train.R, &Xrot);
                }
                else if (ReaderCanReadF32(fallback_reader)) {
                    {
                        const std::lock_guard<std::mutex> lock(fallback_mu);
                        if (!io::ReadDatasetVectorByIdsF32(*fallback_reader, ids, &x_f32, &local_err)) {
                            continue;
                        }
                    }
                    Xrot.rows = train.R.rows;
                    Xrot.cols = x_f32.cols;
                    Xrot.data.resize(static_cast<std::size_t>(Xrot.rows) * static_cast<std::size_t>(Xrot.cols));
                    kernels_local->Gemm(false, false, 1.0f, train.R, x_f32, 0.0f, &Xrot);
                }
                else if (ReaderCanReadF32(fallback_reader)) {
                    {
                        const std::lock_guard<std::mutex> lock(fallback_mu);
                        if (!io::ReadDatasetVectorByIdsF32(*fallback_reader, ids, &x_f32, &local_err)) {
                            continue;
                        }
                    }
                    Xrot.rows = train.R.rows;
                    Xrot.cols = x_f32.cols;
                    Xrot.data.resize(static_cast<std::size_t>(Xrot.rows) * static_cast<std::size_t>(Xrot.cols));
                    kernels_local->Gemm(false, false, 1.0f, train.R, x_f32, 0.0f, &Xrot);
                }
                else {
                    if (!allow_random_fallback) {
                        continue;
                    }
                    if (ReaderCanReadU8(fallback_reader)) {
                        {
                            const std::lock_guard<std::mutex> lock(fallback_mu);
                            if (!io::ReadDatasetVectorByIdsU8(*fallback_reader, ids, &x_u8, &local_err)) {
                                continue;
                            }
                        }
                        kernels_local->ConvertU8ToF32AndRotate(x_u8, train.R, &Xrot);
                    }
                    else {
                        continue;
                    }
                }

                R_full.rows = d;
                R_full.cols = n_real;
                R_full.data.resize(static_cast<std::size_t>(d) * static_cast<std::size_t>(n_real));

                const std::uint8_t* codes_u8 = codes_bytes.data();
                const RootCodeView code0_view = RootCodeView::FromBytes(code0_one_bytes, code0_width_bytes);

                const int n_virt = (m_codes > 0) ? static_cast<int>(virt_coeffs.size()) / m_codes : 0;
                R_virt.rows = d;
                R_virt.cols = n_virt;
                R_virt.data.resize(static_cast<std::size_t>(d) * static_cast<std::size_t>(std::max(0, n_virt)));

                const std::uint8_t* vcodes_u8 = virt_codes_bytes.data();
                for (int v = 0; v < n_virt; ++v) {
                    const std::uint8_t* cu8 = vcodes_u8 + static_cast<std::size_t>(v) * m_codes;
                    const float* a_small = virt_coeffs.data() + static_cast<std::size_t>(v) * m_codes;
                    const float a0v = (v < static_cast<int>(virt_a0.size()))
                                          ? virt_a0[static_cast<std::size_t>(v)]
                                          : 0.0f;
                    AccumulateReconDepth0FromSmallU8(train.C_root, cid, cu8, a_small, a0v, m, d, R_virt.Col(v));
                }

                double cluster_sum = 0.0;
                int valid_cnt = 0;
                for (int i = 0; i < n_real; ++i) {
                    const std::uint8_t* cu8 = codes_u8 + static_cast<std::size_t>(i) * m_codes;
                    const float* a_small = coeffs_small.data() + static_cast<std::size_t>(i) * m_codes;
                    const float a0i = a0[static_cast<std::size_t>(i)];
                    const std::uint32_t p1 = parent[static_cast<std::size_t>(i)];
                    float* dst = R_full.Col(i);
                    if (p1 == 0) {
                        const int code0_root = code0_view.Get(static_cast<std::size_t>(i));
                        AccumulateReconDepth0FromSmallU8(train.C_root, code0_root, cu8, a_small, a0i, m, d, dst);
                    }
                    else {
                        const int code0_root = code0_view.Get(static_cast<std::size_t>(i));
                        AccumulateReconDepthPosFromSmallU8(train.C_root, code0_root, cu8, a_small, a0i, m, d,
                                                           tmp_self.data());
                        const int p = static_cast<int>(p1 - 1);
                        // p >= 0 guaranteed (p1 >= 1). (p-n_virt) >= 0 if p >= n_virt.
                        const float* rp = (p < n_virt)
                                              ? R_virt.Col(p)
                                              : ((p - n_virt < n_real) ? R_full.Col(p - n_virt) : nullptr);
                        if (rp) {
                            for (int r = 0; r < d; ++r) {
                                dst[r] = rp[r] + tmp_self[static_cast<std::size_t>(r)];
                            }
                        }
                        else {
                            LogWarn("ComputeBadMaskInit[cid=" + std::to_string(cid) + "]: parent pr=" +
                                std::to_string(p - n_virt) + " >= n_real=" + std::to_string(n_real) +
                                " at node i=" + std::to_string(i) + "; linkage data corrupt.");
                            std::memcpy(dst, tmp_self.data(), sizeof(float) * static_cast<std::size_t>(d));
                        }
                    }

                    const int pos = FindListPos(gid_pos, real_ids[static_cast<std::size_t>(i)]);
                    if (pos < 0 || pos >= n_real) {
                        LogWarn("ComputeBadMaskInit[cid=" + std::to_string(cid) + "]: gid=" +
                            std::to_string(real_ids[static_cast<std::size_t>(i)]) +
                            " not found in Xrot at node i=" + std::to_string(i) + "; skipping.");
                        continue;
                    }
                    const float* x = Xrot.Col(pos);
                    double e = 0.0;
                    for (int r = 0; r < d; ++r) {
                        const double diff = static_cast<double>(x[r]) - static_cast<double>(dst[r]);
                        e += diff * diff;
                    }
                    cluster_sum += e;
                    ++valid_cnt;

                    ++local_stats.total_cnt;
                    local_stats.sum += e;
                    local_stats.min = std::min(local_stats.min, e);
                    local_stats.max = std::max(local_stats.max, e);
                }

                cluster_mse_sum[static_cast<std::size_t>(cid)] = cluster_sum;
                cluster_cnt[static_cast<std::size_t>(cid)] = valid_cnt;
            }

            thread_stats[static_cast<std::size_t>(tid)] = local_stats;
        }

        if (!ok.load()) {
            if (err) *err = first_open_err;
            return false;
        }

        // Global MSE stats over all real nodes (for logging only).
        std::uint64_t total_cnt = 0;
        double mse_sum = 0.0;
        double mse_min = std::numeric_limits<double>::infinity();
        double mse_max = -std::numeric_limits<double>::infinity();
        for (const auto& ts : thread_stats) {
            total_cnt += ts.total_cnt;
            mse_sum += ts.sum;
            mse_min = std::min(mse_min, ts.min);
            mse_max = std::max(mse_max, ts.max);
        }

        std::vector<double> cluster_avgs;
        cluster_avgs.reserve(static_cast<std::size_t>(nlist));
        for (int c = 0; c < nlist; ++c) {
            if (cluster_cnt[static_cast<std::size_t>(c)] > 0) {
                cluster_avgs.push_back(cluster_mse_sum[static_cast<std::size_t>(c)] /
                    static_cast<double>(cluster_cnt[static_cast<std::size_t>(c)]));
            }
        }

        double baseline = 0.0;
        if (!cluster_avgs.empty()) {
            std::vector<double> sorted_avgs = cluster_avgs;
            std::sort(sorted_avgs.begin(), sorted_avgs.end());
            const double q = std::max(0.0, std::min(1.0, cfg.virtual_cfg.good_fraction));
            if (q <= 0.0) {
                baseline = sorted_avgs.front();
            }
            else if (q >= 1.0) {
                baseline = sorted_avgs.back();
            }
            else {
                const int n_used = static_cast<int>(sorted_avgs.size());
                int pos = static_cast<int>(std::llround(q * static_cast<double>(n_used)));
                pos = std::max(1, std::min(pos, n_used));
                baseline = sorted_avgs[static_cast<std::size_t>(pos - 1)];
            }
        }
        if (baseline_out) {
            *baseline_out = baseline;
        }

        is_bad_cluster_out->assign(static_cast<std::size_t>(nlist), false);
        int bad_cnt = 0;
        for (int c = 0; c < nlist; ++c) {
            if (cluster_cnt[static_cast<std::size_t>(c)] <= 0) {
                continue;
            }
            const double avg = cluster_mse_sum[static_cast<std::size_t>(c)] /
                static_cast<double>(cluster_cnt[static_cast<std::size_t>(c)]);
            if (avg > baseline) {
                (*is_bad_cluster_out)[static_cast<std::size_t>(c)] = true;
                ++bad_cnt;
            }
        }

        if (cfg.train.log_metrics && total_cnt > 0) {
            const double mean = mse_sum / static_cast<double>(total_cnt);
            LogInfo("Train init-linkage MSE (C_root only): Max=" + FormatDoubleLocal(mse_max, 4) +
                ", Min=" + FormatDoubleLocal(mse_min, 4) +
                ", Mean=" + FormatDoubleLocal(mean, 4));
        }

        LogInfo("Virtual init (init linkage): num bad clusters = " + std::to_string(bad_cnt) +
            " / " + std::to_string(nlist) +
            " (baseline=" + std::to_string(baseline) + ")");

        return true;
    }
} // namespace stlq
