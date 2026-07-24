#include "stlq/pipeline/app_utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

#include <omp.h>

#include "stlq/core/blas.h"
#include "stlq/core/threading.h"
#include "stlq/eval/eval_metrics.h"
#include "stlq/eval/recall_linkage_disk.h"
#include "stlq/io/linkage_list_store.h"
#include "stlq/common/logger.h"
#include "stlq/quantizer/linear_algebra.h"

namespace stlq {

namespace {

std::uint64_t MixArchiveU64(std::uint64_t h, std::uint64_t v) {
    h ^= v;
    h ^= h >> 30;
    h *= 0xbf58476d1ce4e5b9ULL;
    h ^= h >> 27;
    h *= 0x94d049bb133111ebULL;
    h ^= h >> 31;
    return h;
}

std::uint64_t MixArchiveString(std::uint64_t h, const std::string& s) {
    for (char c : s) {
        h = MixArchiveU64(h, static_cast<std::uint64_t>(static_cast<unsigned char>(c)));
    }
    return MixArchiveU64(h, s.size());
}

std::uint64_t FloatBits64(double v) {
    static_assert(sizeof(double) == sizeof(std::uint64_t));
    std::uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    return bits;
}

template <typename T>
void AppendArchiveScalar(std::ostream& out, const char* key, const T& value) {
    out << "# " << key << "=" << value << "\n";
}

void AppendArchiveCoeffCodecConfig(std::ostream& out, const Config& cfg, bool use_coeff_codec) {
    if (!use_coeff_codec) return;
    AppendArchiveScalar(out, "coeff_codec_enabled", cfg.large.linkage_coeff_codec.enabled ? 1 : 0);
    AppendArchiveScalar(out, "coeff_codec_granularity", cfg.large.linkage_coeff_codec.granularity);
    out << "# coeff_codec_bits_per_layer=[";
    for (std::size_t i = 0; i < cfg.large.linkage_coeff_codec.bits_per_layer.size(); ++i) {
        if (i) out << ",";
        out << cfg.large.linkage_coeff_codec.bits_per_layer[i];
    }
    out << "]\n";
    out << "# coeff_codec_p_first_candidates=[";
    for (std::size_t i = 0; i < cfg.large.linkage_coeff_codec.p_first_candidates.size(); ++i) {
        if (i) out << ",";
        out << cfg.large.linkage_coeff_codec.p_first_candidates[i];
    }
    out << "]\n";
    out << "# coeff_codec_p_rest_candidates=[";
    for (std::size_t i = 0; i < cfg.large.linkage_coeff_codec.p_rest_candidates.size(); ++i) {
        if (i) out << ",";
        out << cfg.large.linkage_coeff_codec.p_rest_candidates[i];
    }
    out << "]\n";
    AppendArchiveScalar(out, "coeff_codec_use_weighted_quantile",
                        cfg.large.linkage_coeff_codec.use_weighted_quantile ? 1 : 0);
    AppendArchiveScalar(out, "coeff_codec_allow_clip", cfg.large.linkage_coeff_codec.allow_clip ? 1 : 0);
    AppendArchiveScalar(out, "coeff_codec_fit_scale", cfg.large.linkage_coeff_codec.fit_scale ? 1 : 0);
    AppendArchiveScalar(out, "coeff_codec_q_refine_sweeps", cfg.large.linkage_coeff_codec.q_refine_sweeps);
    AppendArchiveScalar(out, "coeff_codec_q_refine_max_layer", cfg.large.linkage_coeff_codec.q_refine_max_layer);
    AppendArchiveScalar(out, "coeff_codec_q_refine_step_limit", cfg.large.linkage_coeff_codec.q_refine_step_limit);
}

bool EndsWithMSuffix(const std::string& s) {
    // Accept stems that end with:
    //   *_m<digits>
    //   *_m<digits>_<YYYYMMDD>
    //   *_m<digits>_<YYYYMMDD>_<seq>
    // This allows raw-load of C++ dated artifacts without double-appending _m.
    std::string stem = s;
    auto strip_suffix_digits = [&](std::size_t ndigits) -> bool {
        if (stem.size() < ndigits + 1) {
            return false;
        }
        const std::size_t start = stem.size() - ndigits;
        if (stem[start - 1] != '_') {
            return false;
        }
        for (std::size_t i = start; i < stem.size(); ++i) {
            const char c = stem[i];
            if (c < '0' || c > '9') {
                return false;
            }
        }
        stem.resize(start - 1);
        return true;
    };

    // Optional seq then optional date.
    (void)strip_suffix_digits(3);
    (void)strip_suffix_digits(8);

    const std::size_t pos = stem.rfind("_m");
    if (pos == std::string::npos || pos + 2 >= stem.size()) {
        return false;
    }
    for (std::size_t i = pos + 2; i < stem.size(); ++i) {
        const char c = stem[i];
        if (c < '0' || c > '9') {
            return false;
        }
    }
    return true;
}

void AppendMSuffixToPath(std::string* path, int m) {
    if (!path || path->empty()) {
        return;
    }
    std::filesystem::path p(*path);
    const std::filesystem::path dir = p.parent_path();
    const std::string stem = p.stem().string();
    const std::string ext = p.extension().string();
    if (stem.empty() || EndsWithMSuffix(stem)) {
        return;
    }
    const std::string new_stem = stem + "_m" + std::to_string(m);
    const std::filesystem::path out =
        dir.empty() ? std::filesystem::path(new_stem + ext) : (dir / (new_stem + ext));
    *path = out.string();
}

}  // namespace

std::string FormatFloat(float value, int precision) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

std::vector<float> ComputeRecallCurve(const std::vector<int>& ground_truth,
                                      const ColMajorMatrix<int>& indices) {
    EvalMetricBundle metrics;
    std::string err;
    if (!ComputeEvalMetricBundle(ground_truth, nullptr, indices, EvalMetricMode::kTop1Recall,
                                 std::vector<int>(), &metrics, &err)) {
        return {};
    }
    return metrics.recall_curve;
}

void PrintRecallCurve(const std::vector<float>& recall) {
    EvalMetricBundle metrics;
    metrics.mode = EvalMetricMode::kTop1Recall;
    metrics.recall_curve = recall;
    PrintEvalMetrics(metrics);
}

void PrintEvalMetrics(const EvalMetricBundle& metrics) {
    for (const std::string& line : BuildEvalMetricReportLines(metrics)) {
        LogInfo(line);
    }
}

void NormalizeHVec(Config* config) {
    NormalizeModelHVec(config);
}

std::string DescribeMatrix(const ColMajorMatrix<float>& mat) {
    std::ostringstream oss;
    oss << mat.rows << "x" << mat.cols;
    return oss.str();
}

bool IsIdentityRotation(const ColMajorMatrix<float>& R) {
    const int d = R.rows;
    if (d <= 0 || R.cols != d) {
        return false;
    }
    for (int c = 0; c < d; ++c) {
        for (int r = 0; r < d; ++r) {
            float want = (r == c) ? 1.0f : 0.0f;
            float diff = std::abs(R(r, c) - want);
            if (diff != 0.0f) {
                return false;
            }
        }
    }
    return true;
}

void ApplyRotationInPlace(const ColMajorMatrix<float>& R, ColMajorMatrix<float>* X) {
    if (!X) {
        return;
    }
    if (IsIdentityRotation(R)) {
        return;
    }
    if (R.cols != R.rows || R.rows != X->rows) {
        return;
    }
    ColMajorMatrix<float> X_rot(X->rows, X->cols);
    {
        // Keep consistent with eval GEMM timing discipline: BLAS multi-threaded for the GEMM,
        // and restored to 1 thread afterwards.
        ScopedBlasThreads blas_scope(OmpMaxThreads());
        GemmRaw(/*trans_a=*/false, /*trans_b=*/false,
                /*m=*/X->rows, /*n=*/X->cols, /*k=*/X->rows,
                /*alpha=*/1.0f,
                /*A=*/R.data.data(), /*lda=*/R.rows,
                /*B=*/X->data.data(), /*ldb=*/X->rows,
                /*beta=*/0.0f,
                /*C=*/X_rot.data.data(), /*ldc=*/X_rot.rows);
    }
    *X = std::move(X_rot);
}

double ApplyRotationInPlaceGemmSeconds(const ColMajorMatrix<float>& R, ColMajorMatrix<float>* X) {
    if (!X) {
        return 0.0;
    }
    if (IsIdentityRotation(R)) {
        return 0.0;
    }
    if (R.cols != R.rows || R.rows != X->rows) {
        return 0.0;
    }
    ColMajorMatrix<float> X_rot(X->rows, X->cols);
    double gemm_sec = 0.0;
    {
        ScopedBlasThreads blas_scope(OmpMaxThreads());
        const double t0 = omp_get_wtime();
        GemmRaw(/*trans_a=*/false, /*trans_b=*/false,
                /*m=*/X->rows, /*n=*/X->cols, /*k=*/X->rows,
                /*alpha=*/1.0f,
                /*A=*/R.data.data(), /*lda=*/R.rows,
                /*B=*/X->data.data(), /*ldb=*/X->rows,
                /*beta=*/0.0f,
                /*C=*/X_rot.data.data(), /*ldc=*/X_rot.rows);
        gemm_sec = omp_get_wtime() - t0;
    }
    *X = std::move(X_rot);
    return gemm_sec;
}

void NormalizeIoPrefixWithM(Config* config) {
    if (!config) {
        return;
    }
    const int m = std::max(1, config->model.m);
    std::string& prefix = config->io.pre_fix;
    if (prefix.empty()) {
        prefix = "m" + std::to_string(m);
    } else if (!EndsWithMSuffix(prefix)) {
        prefix += "_m" + std::to_string(m);
    }

    // Non-large HDF5 result paths: make default file names m-specific as well.
    // If the user explicitly sets io.*_file in config, keep it exactly as provided.
    if (!config->io.train_file_set) AppendMSuffixToPath(&config->io.train_file, m);
    if (!config->io.base_file_set) AppendMSuffixToPath(&config->io.base_file, m);
    if (!config->io.linkage_file_set) AppendMSuffixToPath(&config->io.linkage_file, m);
}

bool PrepareLargeWorkspaceDirs(const Config& config,
                               std::filesystem::path* out_root,
                               std::filesystem::path* tmp_root,
                               std::string* error) {
    if (!config.large.enabled) {
        if (out_root) *out_root = std::filesystem::path();
        if (tmp_root) *tmp_root = std::filesystem::path();
        return true;
    }
    const std::filesystem::path root =
        std::filesystem::path(config.large.output_dir) / config.dataset.name / config.io.pre_fix;
    const std::filesystem::path tmp =
        std::filesystem::path(config.large.tmp_dir) / config.dataset.name / config.io.pre_fix;
    std::error_code ec;
    std::filesystem::create_directories(root, ec);
    if (ec) {
        if (error) {
            *error = "Failed to create large.output_dir workspace: " + root.string();
        }
        return false;
    }
    std::filesystem::create_directories(tmp, ec);
    if (ec) {
        if (error) {
            *error = "Failed to create large.tmp_dir workspace: " + tmp.string();
        }
        return false;
    }
    if (out_root) *out_root = root;
    if (tmp_root) *tmp_root = tmp;
    return true;
}

namespace {

bool HasPrefix(const std::string& s, const char* prefix) {
    const std::size_t n = std::char_traits<char>::length(prefix);
    return s.size() >= n && s.compare(0, n, prefix) == 0;
}

bool HasSuffix(const std::string& s, const char* suffix) {
    const std::size_t n = std::char_traits<char>::length(suffix);
    return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

void RemoveAllBestEffort(const std::filesystem::path& p) {
    std::error_code ec;
    if (!std::filesystem::exists(p, ec)) {
        return;
    }
    std::filesystem::remove_all(p, ec);
    if (ec) {
        LogWarn("AutoCleanup: failed to remove " + p.string() + " : " + ec.message());
    }
}

void RemoveFileBestEffort(const std::filesystem::path& p) {
    std::error_code ec;
    if (!std::filesystem::exists(p, ec)) {
        return;
    }
    std::filesystem::remove(p, ec);
    if (ec) {
        LogWarn("AutoCleanup: failed to remove file " + p.string() + " : " + ec.message());
    }
}

void RemoveIfEmptyBestEffort(const std::filesystem::path& p) {
    std::error_code ec;
    if (p.empty() || !std::filesystem::exists(p, ec) || ec) {
        return;
    }
    if (!std::filesystem::is_directory(p, ec) || ec) {
        return;
    }
    const bool empty = std::filesystem::is_empty(p, ec);
    if (ec || !empty) {
        return;
    }
    std::filesystem::remove(p, ec);
    if (ec) {
        LogWarn("AutoCleanup: failed to remove empty dir " + p.string() + " : " + ec.message());
    }
}

}  // namespace

void CleanupEmptyLargeWorkspaceDirs(const std::filesystem::path& out_root,
                                    const std::filesystem::path& tmp_root) {
    RemoveIfEmptyBestEffort(out_root);
    RemoveIfEmptyBestEffort(tmp_root);
}

void CleanupTrainIntermediates(const Config& config,
                               const std::filesystem::path& exp_root) {
    if (!config.large.enabled || !config.large.cleanup.enabled) {
        return;
    }

    RemoveAllBestEffort(exp_root / "train_basic");
    RemoveAllBestEffort(exp_root / "train_ivf");
    RemoveAllBestEffort(exp_root / "train_list");
    RemoveAllBestEffort(exp_root / "train_linkage_init_list");
    RemoveAllBestEffort(exp_root / "train_linkage_list");
}

void CleanupAfterBaseListReady(const Config& config,
                               const std::filesystem::path& base_basic_dir,
                               const std::filesystem::path& base_list_dir) {
    if (!config.large.enabled) return;
    const ResolvedCleanupFlags f = ResolveCleanupFlags(config.large);

    // base_basic payload cleanup.
    if (!f.keep_base_basic) {
        std::error_code ec;
        if (std::filesystem::exists(base_basic_dir, ec)) {
            for (const auto& ent : std::filesystem::directory_iterator(base_basic_dir, ec)) {
                if (ec) break;
                const auto name = ent.path().filename().string();
                const bool is_bucket = HasPrefix(name, "bucket_") && HasSuffix(name, ".bin");
                const bool is_codes_shard = HasPrefix(name, "codes_shard_") && HasSuffix(name, ".bin");
                const bool is_coeffs_shard = HasPrefix(name, "coeffs_shard_") && HasSuffix(name, ".bin");
                if (is_bucket || is_codes_shard || is_coeffs_shard) {
                    RemoveFileBestEffort(ent.path());
                }
            }
            LogInfo("Cleanup: deleted base_basic payload files in " + base_basic_dir.string());
        }
    }

    // base_basic cluster id cleanup (large file; safe to delete after IVF CSR lists are ready).
    if (!f.keep_base_basic_cluster_id) {
        RemoveFileBestEffort(base_basic_dir / "cluster_id.u32");
        LogInfo("Cleanup: deleted base_basic cluster_id.u32 in " + base_basic_dir.string());
    }

    // base_list raw cleanup (keepable separately from base_list itself).
    // IMPORTANT: base.linkage build needs base_list raw vectors. If we delete raw_*.bin here,
    // linkage_list build will fail later. Therefore, only delete raw vectors at Stage 1 when
    // we are not going to build base.linkage in this run.
    if (!f.keep_base_list_raw) {
        if (config.base.linkage.enabled) {
            LogInfo("Cleanup: keep base_list raw vector files for base.linkage build; will defer raw deletion to Stage 2.");
        } else {
            RemoveFileBestEffort(base_list_dir / "raw_u8.bin");
            RemoveFileBestEffort(base_list_dir / "raw_f32.bin");
            LogInfo("Cleanup: deleted base_list raw vector files in " + base_list_dir.string());
        }
    }
}

void CleanupAfterLinkageListReady(const Config& config,
                                const std::filesystem::path& base_basic_dir,
                                const std::filesystem::path& base_list_dir,
                                const std::filesystem::path& linkage_list_dir) {
    if (!config.large.enabled) return;
    const ResolvedCleanupFlags f = ResolveCleanupFlags(config.large);

    // Delete IVF CSR lists in base_basic/ (optional for linkage-only eval).
    if (!f.keep_base_basic_ivf_lists) {
        RemoveFileBestEffort(base_basic_dir / "ivf_offsets.u64");
        RemoveFileBestEffort(base_basic_dir / "ivf_ids.u32");
        LogInfo("Cleanup: deleted base_basic IVF CSR lists (ivf_offsets.u64, ivf_ids.u32) in " +
                base_basic_dir.string());
    }

    // If nothing under base_basic/ is retained anymore, remove the whole directory
    // (including meta.bin/meta.json/hash/checkpoint leftovers) to match base_list cleanup behavior.
    if (!f.keep_base_basic && !f.keep_base_basic_cluster_id && !f.keep_base_basic_ivf_lists) {
        RemoveAllBestEffort(base_basic_dir);
        LogInfo("Cleanup: deleted base_basic directory " + base_basic_dir.string());
    }

    // Delete entire base_list directory.
    if (!f.keep_base_list) {
        RemoveAllBestEffort(base_list_dir);
        LogInfo("Cleanup: deleted base_list directory " + base_list_dir.string());
    } else {
        // If the user wants to keep base_list but drop raw vectors, do it here (Stage 2)
        // after base.linkage build has completed.
        if (!f.keep_base_list_raw) {
            RemoveFileBestEffort(base_list_dir / "raw_u8.bin");
            RemoveFileBestEffort(base_list_dir / "raw_f32.bin");
            LogInfo("Cleanup: deleted base_list raw vector files in " + base_list_dir.string());
        }
    }

    // Delete linkage_list float coefficient files.
    if (!f.keep_linkage_list_f32) {
        RemoveFileBestEffort(linkage_list_dir / "coeffs.f32");
        RemoveFileBestEffort(linkage_list_dir / "a0.f32");
        RemoveFileBestEffort(linkage_list_dir / "virt_coeffs.f32");
        RemoveFileBestEffort(linkage_list_dir / "virt_a0.f32");
        std::string meta_err;
        if (!io::UpdateLinkageListMetaStoreCoeffsF32(linkage_list_dir.string(), /*store_coeffs_f32=*/false, &meta_err)) {
            LogWarn("Cleanup: deleted linkage_list float coeff files but failed to update meta.bin: " + meta_err);
        }
        LogInfo("Cleanup: deleted linkage_list float coeff files in " + linkage_list_dir.string());
    }

    // Delete parent.u32 when LOUDS exists and user doesn't want it.
    if (!f.keep_linkage_parent_u32) {
        const auto louds = linkage_list_dir / "parent_louds.bin";
        const auto parent_u32 = linkage_list_dir / "parent.u32";
        std::error_code ec;
        if (std::filesystem::exists(louds, ec) && std::filesystem::exists(parent_u32, ec)) {
            RemoveFileBestEffort(parent_u32);
            LogInfo("Cleanup: deleted linkage_list/parent.u32 (LOUDS available)");
        }
    }
}

// ---- Eval result archival ----

std::uint64_t ComputeEvalArchiveHash(const Config& cfg, bool use_coeff_codec) {
    // Mix eval-relevant config fields that affect recall or QPS measurement.
    // Uses a simple FNV-1a-style 64-bit mix over a sequence of integers/strings.
    std::uint64_t h = 0x9e3779b97f4a7c15ULL;  // golden ratio seed
    // --- probe count & mode ---
    h = MixArchiveU64(h, static_cast<std::uint64_t>(cfg.eval.linkage_nprobe));
    h = MixArchiveString(h, cfg.eval.disk_norm2_mode);
    h = MixArchiveU64(h, static_cast<std::uint64_t>(cfg.eval.disk_norm2_lut_kmeans_niter));
    h = MixArchiveU64(h, FloatBits64(static_cast<double>(cfg.eval.disk_norm2_lut_log_alpha)));
    h = MixArchiveU64(h, FloatBits64(static_cast<double>(cfg.eval.disk_norm2_lut_piecewise_p1)));
    h = MixArchiveU64(h, FloatBits64(static_cast<double>(cfg.eval.disk_norm2_lut_piecewise_p2)));
    h = MixArchiveU64(h, FloatBits64(static_cast<double>(cfg.eval.disk_norm2_lut_piecewise_count_weight)));
    h = MixArchiveU64(h, FloatBits64(static_cast<double>(cfg.eval.disk_norm2_lut_piecewise_range_weight)));
    h = MixArchiveString(h, cfg.eval.linkage_ivf_probe_mode);
    h = MixArchiveU64(h, static_cast<std::uint64_t>(cfg.eval.linkage_ivf_hier2_top_coarse));
    // Backward-compatibility: only include HNSW knobs when the mode is active.
    // Otherwise old archives (exact/hier2) would become unresolvable by hash.
    if (cfg.eval.linkage_ivf_probe_mode == "hnsw") {
        h = MixArchiveU64(h, static_cast<std::uint64_t>(cfg.eval.linkage_ivf_hnsw_M));
        h = MixArchiveU64(h, static_cast<std::uint64_t>(cfg.eval.linkage_ivf_hnsw_ef_construction));
        h = MixArchiveU64(h, static_cast<std::uint64_t>(cfg.eval.linkage_ivf_hnsw_ef_search));
    }
    h = MixArchiveU64(h, cfg.eval.linkage_gpu_norm_enable ? 1ULL : 0ULL);
    h = MixArchiveU64(h, cfg.eval.linkage_gpu_scan_enable ? 1ULL : 0ULL);
    // --- threading (affects QPS) ---
    h = MixArchiveU64(h, static_cast<std::uint64_t>(cfg.runtime.omp_threads));
    // --- query block (conservative: might affect batching) ---
    h = MixArchiveU64(h, static_cast<std::uint64_t>(cfg.eval.linkage_query_block));
    // --- IVF flags ---
    h = MixArchiveU64(h, cfg.eval.base_use_ivf ? 1ULL : 0ULL);
    h = MixArchiveU64(h, cfg.eval.linkage_use_ivf_disk ? 1ULL : 0ULL);
    // --- bench params ---
    h = MixArchiveU64(h, static_cast<std::uint64_t>(cfg.eval.linkage_warmup));
    h = MixArchiveU64(h, static_cast<std::uint64_t>(cfg.eval.linkage_repeat));
    h = MixArchiveU64(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(cfg.eval.linkage_preload_clusters_io_threads)));
    h = MixArchiveU64(h, static_cast<std::uint64_t>(cfg.eval.linkage_louds_huffman_bench_times));
    h = MixArchiveU64(h, cfg.eval.linkage_parent_adaptive_u16_cache ? 1ULL : 0ULL);
    h = MixArchiveU64(h, cfg.eval.linkage_parent_louds_enable ? 1ULL : 0ULL);
    h = MixArchiveU64(h, cfg.eval.linkage_parent_louds_native_eval ? 1ULL : 0ULL);
    h = MixArchiveU64(h, cfg.eval.parent_louds_build_indices ? 1ULL : 0ULL);
    h = MixArchiveU64(h, static_cast<std::uint64_t>(cfg.eval.parent_louds_select_stride));
    h = MixArchiveU64(h, static_cast<std::uint64_t>(cfg.eval.parent_louds_rank_words_per_super_log2));
    h = MixArchiveU64(h, use_coeff_codec ? 1ULL : 0ULL);
    if (use_coeff_codec) {
        h = MixArchiveU64(h, cfg.large.linkage_coeff_codec.enabled ? 1ULL : 0ULL);
        h = MixArchiveString(h, cfg.large.linkage_coeff_codec.granularity);
        for (int bits : cfg.large.linkage_coeff_codec.bits_per_layer) {
            h = MixArchiveU64(h, static_cast<std::uint64_t>(bits));
        }
        for (double v : cfg.large.linkage_coeff_codec.p_first_candidates) {
            h = MixArchiveU64(h, FloatBits64(v));
        }
        for (double v : cfg.large.linkage_coeff_codec.p_rest_candidates) {
            h = MixArchiveU64(h, FloatBits64(v));
        }
        h = MixArchiveU64(h, cfg.large.linkage_coeff_codec.use_weighted_quantile ? 1ULL : 0ULL);
        h = MixArchiveU64(h, cfg.large.linkage_coeff_codec.allow_clip ? 1ULL : 0ULL);
        h = MixArchiveU64(h, cfg.large.linkage_coeff_codec.fit_scale ? 1ULL : 0ULL);
        h = MixArchiveU64(h, static_cast<std::uint64_t>(cfg.large.linkage_coeff_codec.q_refine_sweeps));
        h = MixArchiveU64(h, static_cast<std::uint64_t>(cfg.large.linkage_coeff_codec.q_refine_max_layer));
        h = MixArchiveU64(h, static_cast<std::uint64_t>(cfg.large.linkage_coeff_codec.q_refine_step_limit));
    }
    return h;
}

std::uint64_t ComputeEvalMetricsArchiveHash(const Config& cfg, EvalMetricMode mode, bool use_coeff_codec) {
    std::uint64_t h = ComputeEvalArchiveHash(cfg, use_coeff_codec);
    h = MixArchiveU64(h, static_cast<std::uint64_t>(EvalMetricModeToInt(mode)));
    for (int v : cfg.eval.metric_gt_topks) {
        h = MixArchiveU64(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(v)));
    }
    return h;
}

bool ArchiveEvalResult(const std::string& run_root,
                       const std::string& label,
                       const std::vector<float>& recall,
                       int nq,
                       std::uint64_t eval_hash,
                       bool use_coeff_codec,
                       const Config& cfg,
                       const DiskLinkageEvalTiming* timing,
                       const DiskLinkageEvalSession* session,
                       const std::vector<double>& repeat_core_secs,
                       const std::vector<double>& repeat_qps_vals,
                       std::string* err) {
    namespace fs = std::filesystem;
    const fs::path eval_dir = fs::path(run_root) / "eval_result";
    std::error_code ec;
    fs::create_directories(eval_dir, ec);
    if (ec) {
        if (err) *err = "ArchiveEvalResult: cannot create dir: " + eval_dir.string();
        return false;
    }

    // Filename: recall_<label>_0x<hash16>.txt
    std::ostringstream fname;
    fname << "recall_" << label << "_0x"
          << std::hex << std::setfill('0') << std::setw(16) << eval_hash
          << ".txt";
    const fs::path out_path = eval_dir / fname.str();

    std::ofstream out(out_path);
    if (!out) {
        if (err) *err = "ArchiveEvalResult: cannot open " + out_path.string();
        return false;
    }

    // Recall curve: one value per line, k=1..topk, percent with 2 decimals, pure numbers.
    // Format: "XX.YY\n" — CSV-importable, no labels.
    const int topk = static_cast<int>(recall.size());
    for (int k = 1; k <= topk; ++k) {
        out << std::fixed << std::setprecision(2)
            << (100.0f * recall[static_cast<std::size_t>(k - 1)]) << "\n";
    }

    // Timing/bench summary (separated from recall data by blank line + comment header).
    out << "\n";
    out << "# nq=" << nq << "\n";
    // Hash-relevant eval parameters (these determine the filename hash).
    out << "# nprobe=" << cfg.eval.linkage_nprobe << "\n";
    out << "# coeff_mode=" << (use_coeff_codec ? "int8" : "float") << "\n";
    out << "# disk_norm2_mode=" << cfg.eval.disk_norm2_mode << "\n";
    out << "# disk_norm2_lut_kmeans_niter=" << cfg.eval.disk_norm2_lut_kmeans_niter << "\n";
    out << "# disk_norm2_lut_log_alpha=" << cfg.eval.disk_norm2_lut_log_alpha << "\n";
    out << "# disk_norm2_lut_piecewise_p1=" << cfg.eval.disk_norm2_lut_piecewise_p1 << "\n";
    out << "# disk_norm2_lut_piecewise_p2=" << cfg.eval.disk_norm2_lut_piecewise_p2 << "\n";
    out << "# disk_norm2_lut_piecewise_count_weight=" << cfg.eval.disk_norm2_lut_piecewise_count_weight << "\n";
    out << "# disk_norm2_lut_piecewise_range_weight=" << cfg.eval.disk_norm2_lut_piecewise_range_weight << "\n";
    out << "# ivf_probe_mode=" << cfg.eval.linkage_ivf_probe_mode << "\n";
    out << "# ivf_hier2_top_coarse=" << cfg.eval.linkage_ivf_hier2_top_coarse << "\n";
    out << "# ivf_hnsw_M=" << cfg.eval.linkage_ivf_hnsw_M << "\n";
    out << "# ivf_hnsw_ef_construction=" << cfg.eval.linkage_ivf_hnsw_ef_construction << "\n";
    out << "# ivf_hnsw_ef_search=" << cfg.eval.linkage_ivf_hnsw_ef_search << "\n";
    out << "# linkage_gpu_norm_enable=" << (cfg.eval.linkage_gpu_norm_enable ? 1 : 0) << "\n";
    out << "# gpu_scan_enable=" << (cfg.eval.linkage_gpu_scan_enable ? 1 : 0) << "\n";
    out << "# omp_threads=" << cfg.runtime.omp_threads << "\n";
    out << "# query_block=" << cfg.eval.linkage_query_block << "\n";
    out << "# warmup=" << cfg.eval.linkage_warmup << "\n";
    out << "# repeat=" << cfg.eval.linkage_repeat << "\n";
    out << "# preload_io_threads=" << cfg.eval.linkage_preload_clusters_io_threads << "\n";
    out << "# louds_huffman_bench_times=" << cfg.eval.linkage_louds_huffman_bench_times << "\n";
    out << "# parent_adaptive_u16_cache=" << (cfg.eval.linkage_parent_adaptive_u16_cache ? 1 : 0) << "\n";
    out << "# parent_louds_enable=" << (cfg.eval.linkage_parent_louds_enable ? 1 : 0) << "\n";
    out << "# parent_louds_native_eval=" << (cfg.eval.linkage_parent_louds_native_eval ? 1 : 0) << "\n";
    out << "# parent_louds_build_indices=" << (cfg.eval.parent_louds_build_indices ? 1 : 0) << "\n";
    out << "# parent_louds_select_stride=" << cfg.eval.parent_louds_select_stride << "\n";
    out << "# parent_louds_rank_words_per_super_log2=" << cfg.eval.parent_louds_rank_words_per_super_log2 << "\n";
    AppendArchiveCoeffCodecConfig(out, cfg, use_coeff_codec);
    if (timing) {
        out << "# core_sec=" << std::fixed << std::setprecision(6) << timing->core_wall_sec << "\n";
        out << "# qt_gemm_sec=" << std::fixed << std::setprecision(6) << timing->qt_gemm_wall_sec << "\n";
        out << "# qt_rotate_sec=" << std::fixed << std::setprecision(6) << timing->qt_rotate_wall_sec << "\n";
        out << "# qt_coarse_gemm_sec=" << std::fixed << std::setprecision(6) << timing->qt_coarse_gemm_wall_sec << "\n";
        out << "# qt_root_small_gemm_sec=" << std::fixed << std::setprecision(6) << timing->qt_root_small_gemm_wall_sec << "\n";
        out << "# qt_one_gemm_sec=" << std::fixed << std::setprecision(6) << timing->qt_one_gemm_wall_sec << "\n";
        out << "# qt_other_sec=" << std::fixed << std::setprecision(6) << timing->qt_other_wall_sec << "\n";
        out << "# probe_sel_sec=" << std::fixed << std::setprecision(6) << timing->probe_sel_wall_sec << "\n";
        out << "# scan_topk_sec=" << std::fixed << std::setprecision(6) << timing->scan_topk_wall_sec << "\n";
        out << "# louds_sec=" << std::fixed << std::setprecision(6) << timing->louds_wall_sec << "\n";
        out << "# louds_cpu_sec=" << std::fixed << std::setprecision(6) << timing->louds_cpu_sec << "\n";
        out << "# huffman_sec=" << std::fixed << std::setprecision(6) << timing->huffman_wall_sec << "\n";
        out << "# huffman_cpu_sec=" << std::fixed << std::setprecision(6) << timing->huffman_cpu_sec << "\n";
        if (session) {
            out << "# louds_load_mode=" << session->louds_load_mode << "\n";
            out << "# louds_index_enabled=" << session->louds_index_effective_enabled << "\n";
            out << "# louds_index_select_stride=" << session->louds_index_select_stride << "\n";
            out << "# louds_index_rank_words_per_super_log2=" << session->louds_index_rank_words_per_super_log2 << "\n";
            out << "# louds_index_rank_bits_per_real=" << std::fixed << std::setprecision(6) << session->louds_index_rank_bits_per_real << "\n";
            out << "# louds_index_select_bits_per_real=" << std::fixed << std::setprecision(6) << session->louds_index_select_bits_per_real << "\n";
            out << "# louds_index_total_bits_per_real=" << std::fixed << std::setprecision(6) << session->louds_index_total_bits_per_real << "\n";
        }
    }

    // Per-repeat results and aggregate stats.
    const int nrep = static_cast<int>(repeat_core_secs.size());
    if (nrep > 0) {
        for (int i = 0; i < nrep; ++i) {
            out << "# run=" << i
                << " core_sec=" << std::fixed << std::setprecision(6) << repeat_core_secs[static_cast<std::size_t>(i)]
                << " qps_core=" << std::fixed << std::setprecision(1) << repeat_qps_vals[static_cast<std::size_t>(i)];
            out << "\n";
        }
        // Core: median + avg.
        std::vector<double> sorted_core = repeat_core_secs;
        std::sort(sorted_core.begin(), sorted_core.end());
        const double med_core = sorted_core[sorted_core.size() / 2];
        const double med_qps_core = (med_core > 0.0) ? (static_cast<double>(nq) / med_core) : 0.0;
        double sum_core = 0.0;
        for (double v : repeat_core_secs) sum_core += v;
        const double avg_core = sum_core / static_cast<double>(nrep);
        const double avg_qps_core = (avg_core > 0.0) ? (static_cast<double>(nq) / avg_core) : 0.0;
        out << "# repeat=" << nrep << "\n";
        out << "# median_core_sec=" << std::fixed << std::setprecision(6) << med_core << "\n";
        out << "# median_qps_core=" << std::fixed << std::setprecision(1) << med_qps_core << "\n";
        out << "# avg_core_sec=" << std::fixed << std::setprecision(6) << avg_core << "\n";
        out << "# avg_qps_core=" << std::fixed << std::setprecision(1) << avg_qps_core << "\n";
    }

    out.flush();
    if (!out) {
        if (err) *err = "ArchiveEvalResult: write error: " + out_path.string();
        return false;
    }

    LogInfo("Archived eval result: " + out_path.string());
    return true;
}

bool ArchiveEvalMetricsResult(const std::string& run_root,
                              const std::string& label,
                              const EvalMetricBundle& metrics,
                              int nq,
                              std::uint64_t eval_hash,
                              bool use_coeff_codec,
                              const Config& cfg,
                              const DiskLinkageEvalTiming* timing,
                              const DiskLinkageEvalSession* session,
                              const std::vector<double>& repeat_core_secs,
                              const std::vector<double>& repeat_qps_vals,
                              std::string* err) {
    namespace fs = std::filesystem;
    (void)timing;
    (void)repeat_core_secs;
    (void)repeat_qps_vals;
    const fs::path eval_dir = fs::path(run_root) / "eval_result";
    std::error_code ec;
    fs::create_directories(eval_dir, ec);
    if (ec) {
        if (err) *err = "ArchiveEvalMetricsResult: cannot create dir: " + eval_dir.string();
        return false;
    }

    std::ostringstream fname;
    fname << "metrics_" << label << "_0x"
          << std::hex << std::setfill('0') << std::setw(16) << eval_hash
          << ".txt";
    const fs::path out_path = eval_dir / fname.str();
    std::ofstream out(out_path);
    if (!out) {
        if (err) *err = "ArchiveEvalMetricsResult: cannot open " + out_path.string();
        return false;
    }

    auto write_curve = [&](const char* header, const std::vector<float>& curve) {
        out << "# section=" << header << "\n";
        for (float v : curve) {
            out << std::fixed << std::setprecision(2) << (100.0f * v) << "\n";
        }
        out << "\n";
    };
    auto write_curve_from = [&](const char* header, const std::vector<float>& curve, int start_k) {
        out << "# section=" << header << "\n";
        out << "# start_k=" << start_k << "\n";
        for (int k = std::max(1, start_k); k <= static_cast<int>(curve.size()); ++k) {
            out << std::fixed << std::setprecision(2)
                << (100.0f * curve[static_cast<std::size_t>(k - 1)]) << "\n";
        }
        out << "\n";
    };
    auto write_ndcg_curve_from = [&](const char* header, const std::vector<float>& curve, int start_k) {
        out << "# section=" << header << "\n";
        out << "# start_k=" << start_k << "\n";
        for (int k = std::max(1, start_k); k <= static_cast<int>(curve.size()); ++k) {
            out << std::fixed << std::setprecision(6)
                << curve[static_cast<std::size_t>(k - 1)] << "\n";
        }
        out << "\n";
    };
    write_curve("top1_recall_percent", metrics.recall_curve);
    for (const EvalGtMetricCurve& curve : metrics.gt_curves) {
        std::ostringstream hdr;
        hdr << "top" << curve.gt_k << "_recall_percent";
        write_curve_from(hdr.str().c_str(), curve.recall_curve, curve.gt_k);
        if (metrics.mode == EvalMetricMode::kTopKNdcg) {
            std::ostringstream ndcg_hdr;
            ndcg_hdr << "ndcg" << curve.gt_k << "_raw";
            write_ndcg_curve_from(ndcg_hdr.str().c_str(), curve.ndcg_curve, curve.gt_k);
        }
    }
    out << "# metric_mode=" << EvalMetricModeToInt(metrics.mode) << "\n";
    out << "# metric_gt_topks=[";
    for (std::size_t i = 0; i < metrics.gt_curves.size(); ++i) {
        if (i) out << ",";
        out << metrics.gt_curves[i].gt_k;
    }
    out << "]\n";
    out << "# metric_summary=" << FormatEvalMetricSummary(metrics) << "\n";
    out << "# nq=" << nq << "\n";
    out << "# nprobe=" << cfg.eval.linkage_nprobe << "\n";
    out << "# coeff_mode=" << (use_coeff_codec ? "int8" : "float") << "\n";
    out << "# disk_norm2_mode=" << cfg.eval.disk_norm2_mode << "\n";
    out << "# disk_norm2_lut_kmeans_niter=" << cfg.eval.disk_norm2_lut_kmeans_niter << "\n";
    out << "# disk_norm2_lut_log_alpha=" << cfg.eval.disk_norm2_lut_log_alpha << "\n";
    out << "# disk_norm2_lut_piecewise_p1=" << cfg.eval.disk_norm2_lut_piecewise_p1 << "\n";
    out << "# disk_norm2_lut_piecewise_p2=" << cfg.eval.disk_norm2_lut_piecewise_p2 << "\n";
    out << "# disk_norm2_lut_piecewise_count_weight=" << cfg.eval.disk_norm2_lut_piecewise_count_weight << "\n";
    out << "# disk_norm2_lut_piecewise_range_weight=" << cfg.eval.disk_norm2_lut_piecewise_range_weight << "\n";
    AppendArchiveCoeffCodecConfig(out, cfg, use_coeff_codec);
    if (session) {
        out << "# louds_load_mode=" << session->louds_load_mode << "\n";
    }
    out.flush();
    if (!out) {
        if (err) *err = "ArchiveEvalMetricsResult: write error: " + out_path.string();
        return false;
    }
    LogInfo("Archived eval metrics: " + out_path.string());
    return true;
}

}  // namespace stlq
