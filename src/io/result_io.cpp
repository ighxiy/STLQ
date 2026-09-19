#include "stlq/io/result_io.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cctype>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <fstream>
#include <ios>
#include <sstream>
#include <string>

#if defined(STLQ_ENABLE_HDF5) && STLQ_ENABLE_HDF5
#include "stlq/io/hdf5_io.h"
#endif
#include "stlq/common/logger.h"

namespace stlq::io {

namespace {

bool EnsureParentDir(const std::string& path, std::string* error) {
    std::filesystem::path p(path);
    std::filesystem::path dir = p.parent_path();
    if (dir.empty()) {
        return true;
    }
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        if (error) {
            *error = "Failed to create output directory: " + dir.string();
        }
        return false;
    }
    return true;
}

std::string TodayYYYYMMDD() {
    const auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::setfill('0') << std::setw(4) << (tm.tm_year + 1900)
        << std::setw(2) << (tm.tm_mon + 1)
        << std::setw(2) << tm.tm_mday;
    return oss.str();
}

std::string DumpConfigMini(const Config& config) {
    std::ostringstream oss;
    oss << "dataset.name = \"" << config.dataset.name << "\"\n";
    oss << "dataset.data_root = \"" << config.dataset.data_root << "\"\n";
    oss << "dataset.groundtruth_add1 = " << config.dataset.groundtruth_add1 << "\n";
    oss << "dataset.ntrain = " << config.dataset.ntrain << "\n";
    oss << "dataset.nbase = " << config.dataset.nbase << "\n";
    oss << "dataset.nquery = " << config.dataset.nquery << "\n";
    oss << "dataset.k = " << config.dataset.k << "\n";
    oss << "model.m = " << config.model.m << "\n";
    oss << "model.h_vec = [";
    for (std::size_t i = 0; i < config.model.h_vec.size(); ++i) {
        if (i) {
            oss << ", ";
        }
        oss << config.model.h_vec[i];
    }
    oss << "]\n";
    oss << "advanced.eval_only = " << (config.advanced.eval_only ? "true" : "false") << "\n";
    oss << "runtime.omp_threads = " << config.runtime.omp_threads << "\n";
    oss << "train.enabled = " << (config.train.enabled ? "true" : "false") << "\n";
    oss << "train.use_opq_rotation = " << (config.train.use_opq_rotation ? "true" : "false") << "\n";
    oss << "train.max_R_iters = " << config.train.max_R_iters << "\n";
    oss << "train.ils_iters = " << config.train.ils_iters << "\n";
    oss << "train.icm_iters = " << config.train.icm_iters << "\n";
    oss << "train.perturb_k = " << config.train.perturb_k << "\n";
    oss << "train.kmeans_iters = " << config.train.kmeans_iters << "\n";
    oss << "train.kmeans_tol = " << config.train.kmeans_tol << "\n";
    oss << "train.kmeans_init = \"" << config.train.kmeans_init << "\"\n";
    oss << "train.init_samples = " << config.train.init_samples << "\n";
    oss << "train.kmeans_initial_weight = " << config.train.kmeans_initial_weight << "\n";
    oss << "train.kmeans_min_weight = " << config.train.kmeans_min_weight << "\n";
    oss << "train.kmeans_outlier_quantile = " << config.train.kmeans_outlier_quantile << "\n";
    oss << "train.kmeans_cost_threshold = " << config.train.kmeans_cost_threshold << "\n";
    oss << "train.kmeans_annealing_factor = " << config.train.kmeans_annealing_factor << "\n";
    oss << "train.kmeans_warmup_iters = " << config.train.kmeans_warmup_iters << "\n";
    oss << "train.seed = " << config.train.seed << "\n";

    oss << "train.linkage.enabled = " << (config.train.linkage.enabled ? "true" : "false") << "\n";
    oss << "train.linkage.reference_policy = \"" << config.train.linkage.reference_policy << "\"\n";
    oss << "train.linkage.root_percentile = " << config.train.linkage.root_percentile << "\n";
    oss << "train.linkage.num_layers = " << config.train.linkage.num_layers << "\n";
    oss << "train.linkage.max_depth = " << config.train.linkage.max_depth << "\n";
    oss << "train.linkage.knn_k = " << config.train.linkage.knn_k << "\n";
    oss << "train.linkage.depth_k = " << config.train.linkage.depth_k << "\n";
    oss << "train.linkage.icm_round = " << config.train.linkage.icm_round << "\n";
    oss << "train.linkage.use_ils = " << (config.train.linkage.use_ils ? "true" : "false") << "\n";
    oss << "train.linkage.ils_rounds = " << config.train.linkage.ils_rounds << "\n";
    oss << "train.linkage.ils_perturb_layers = " << config.train.linkage.ils_perturb_layers << "\n";
    oss << "train.linkage.seed = " << config.train.linkage.seed << "\n";
    oss << "train.init_linkage_icm_round = " << config.train.init_linkage_icm_round << "\n";
    oss << "train.init_linkage_ils_rounds = " << config.train.init_linkage_ils_rounds << "\n";
    oss << "train.init_linkage_mode = \"" << config.train.init_linkage_mode << "\"\n";

    oss << "base.encode.enabled = " << (config.base.encode.enabled ? "true" : "false") << "\n";
    oss << "base.encode.use_abs = " << (config.base.encode.use_abs ? "true" : "false") << "\n";
    oss << "base.encode.H_beam = " << config.base.encode.H_beam << "\n";
    oss << "base.encode.ils_iters = " << config.base.encode.ils_iters << "\n";
    oss << "base.encode.icm_iters = " << config.base.encode.icm_iters << "\n";
    oss << "base.encode.perturb_k = " << config.base.encode.perturb_k << "\n";
    oss << "base.encode.hnorms = " << config.base.encode.hnorms << "\n";
    oss << "base.encode.seed = " << config.base.encode.seed << "\n";

    oss << "base.linkage.enabled = " << (config.base.linkage.enabled ? "true" : "false") << "\n";
    oss << "base.linkage.reference_policy = \"" << config.base.linkage.reference_policy << "\"\n";
    oss << "base.linkage.root_percentile = " << config.base.linkage.root_percentile << "\n";
    oss << "base.linkage.num_layers = " << config.base.linkage.num_layers << "\n";
    oss << "base.linkage.max_depth = " << config.base.linkage.max_depth << "\n";
    oss << "base.linkage.knn_k = " << config.base.linkage.knn_k << "\n";
    oss << "base.linkage.depth_k = " << config.base.linkage.depth_k << "\n";
    oss << "base.linkage.icm_round = " << config.base.linkage.icm_round << "\n";
    oss << "base.linkage.use_ils = " << (config.base.linkage.use_ils ? "true" : "false") << "\n";
    oss << "base.linkage.ils_rounds = " << config.base.linkage.ils_rounds << "\n";
    oss << "base.linkage.ils_perturb_layers = " << config.base.linkage.ils_perturb_layers << "\n";
    oss << "base.linkage.seed = " << config.base.linkage.seed << "\n";

    oss << "virtual.enabled = " << (config.virtual_cfg.enabled ? "true" : "false") << "\n";
    oss << "virtual.anchor_policy = \"" << config.virtual_cfg.anchor_policy << "\"\n";
    oss << "virtual.subkmeans_iters = " << config.virtual_cfg.subkmeans_iters << "\n";
    oss << "virtual.virtual_ratio = " << config.virtual_cfg.virtual_ratio << "\n";
    oss << "virtual.good_fraction = " << config.virtual_cfg.good_fraction << "\n";
    oss << "virtual.min_virtual = " << config.virtual_cfg.min_virtual << "\n";
    oss << "virtual.max_virtual = " << config.virtual_cfg.max_virtual << "\n";
    oss << "virtual.alpha_bad = " << config.virtual_cfg.alpha_bad << "\n";
    oss << "virtual.use_fixed_virtual_per_cluster = " << (config.virtual_cfg.use_fixed_virtual_per_cluster ? "true" : "false") << "\n";
    oss << "virtual.fixed_virtual_per_cluster = " << config.virtual_cfg.fixed_virtual_per_cluster << "\n";
    oss << "virtual.umap_knn_k = " << config.virtual_cfg.umap_knn_k << "\n";
    oss << "virtual.local_connectivity = " << config.virtual_cfg.local_connectivity << "\n";
    oss << "virtual.overlap_thr = " << config.virtual_cfg.overlap_thr << "\n";
    oss << "virtual.prefer_peaks = " << (config.virtual_cfg.prefer_peaks ? "true" : "false") << "\n";
    oss << "virtual.anchor_neighbor_k = " << config.virtual_cfg.anchor_neighbor_k << "\n";

    oss << "hnsw.M = " << config.hnsw.M << "\n";
    oss << "hnsw.candidate_multiplier_good = " << config.hnsw.candidate_multiplier_good << "\n";
    oss << "hnsw.candidate_multiplier_bad = " << config.hnsw.candidate_multiplier_bad << "\n";
    oss << "hnsw.ef_construction_cap = " << config.hnsw.ef_construction_cap << "\n";

    oss << "io.pre_fix = \"" << config.io.pre_fix << "\"\n";
    oss << "io.config_file = \"" << config.io.config_file << "\"\n";
    oss << "io.train_file = \"" << config.io.train_file << "\"\n";
    oss << "io.base_file = \"" << config.io.base_file << "\"\n";
    oss << "io.linkage_file = \"" << config.io.linkage_file << "\"\n";
    oss << "io.save_train = " << (config.io.save_train ? "true" : "false") << "\n";
    oss << "io.save_base = " << (config.io.save_base ? "true" : "false") << "\n";
    oss << "io.save_linkage = " << (config.io.save_linkage ? "true" : "false") << "\n";
    oss << "io.result_format = \"" << config.io.result_format << "\"\n";
    oss << "io.hdf5_layout = \"" << config.io.hdf5_layout << "\"\n";
    oss << "io.load_date = \"" << config.io.load_date << "\"\n";
    oss << "io.load_seq = \"" << config.io.load_seq << "\"\n";

    oss << "eval.base_use_ivf = " << (config.eval.base_use_ivf ? "true" : "false") << "\n";
    oss << "eval.base_nprobe = " << config.eval.base_nprobe << "\n";
    return oss.str();
}

std::string NowLocalISO8601() {
    const auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::setfill('0')
        << std::setw(4) << (tm.tm_year + 1900) << "-"
        << std::setw(2) << (tm.tm_mon + 1) << "-"
        << std::setw(2) << tm.tm_mday << "T"
        << std::setw(2) << tm.tm_hour << ":"
        << std::setw(2) << tm.tm_min << ":"
        << std::setw(2) << tm.tm_sec;
    return oss.str();
}

std::vector<std::uint8_t> BoolToU8(const std::vector<bool>& vec) {
    std::vector<std::uint8_t> out(vec.size());
    for (std::size_t i = 0; i < vec.size(); ++i) {
        out[i] = vec[i] ? static_cast<std::uint8_t>(1) : static_cast<std::uint8_t>(0);
    }
    return out;
}

std::vector<bool> U8ToBool(const std::vector<std::uint8_t>& vec) {
    std::vector<bool> out;
    out.resize(vec.size());
    for (std::size_t i = 0; i < vec.size(); ++i) {
        out[i] = (vec[i] != 0);
    }
    return out;
}

enum class TrainResultStorageFormat {
    kHdf5,
    kBin,
};

using ResultStorageFormat = TrainResultStorageFormat;

bool HasExtension(const std::string& path, const char* ext) {
    std::filesystem::path p(path);
    std::string got = p.extension().string();
    std::transform(got.begin(), got.end(), got.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return got == ext;
}

bool FileStartsWithMagic(const std::string& path, const char* magic, std::size_t magic_len) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        return false;
    }
    std::string got(magic_len, '\0');
    in.read(got.data(), static_cast<std::streamsize>(magic_len));
    return in.gcount() == static_cast<std::streamsize>(magic_len) &&
           std::memcmp(got.data(), magic, magic_len) == 0;
}

bool IsNativeBinPathOrFile(const std::string& path) {
    if (HasExtension(path, ".stlqbin") || HasExtension(path, ".bin")) {
        return true;
    }
    return FileStartsWithMagic(path, "STLQTRN1", 8) ||
           FileStartsWithMagic(path, "STLQBAS1", 8) ||
           FileStartsWithMagic(path, "STLQLNK1", 8);
}

TrainResultStorageFormat ChooseTrainResultSaveFormat(const Config& config, const std::string& path) {
    if (config.io.result_format == "bin") {
        return TrainResultStorageFormat::kBin;
    }
    if (config.io.result_format == "hdf5") {
        return TrainResultStorageFormat::kHdf5;
    }
    if (HasExtension(path, ".stlqbin") || HasExtension(path, ".bin")) {
        return TrainResultStorageFormat::kBin;
    }
#if defined(STLQ_ENABLE_HDF5) && STLQ_ENABLE_HDF5
    return TrainResultStorageFormat::kHdf5;
#else
    return TrainResultStorageFormat::kBin;
#endif
}

ResultStorageFormat ChooseResultSaveFormat(const Config& config, const std::string& path) {
    return ChooseTrainResultSaveFormat(config, path);
}

TrainResultStorageFormat ChooseTrainResultLoadFormat(const std::string& path) {
    if (IsNativeBinPathOrFile(path)) {
        return TrainResultStorageFormat::kBin;
    }
    return TrainResultStorageFormat::kHdf5;
}

ResultStorageFormat ChooseResultLoadFormat(const std::string& path) {
    return ChooseTrainResultLoadFormat(path);
}

template <typename T>
bool WritePod(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), static_cast<std::streamsize>(sizeof(T)));
    return static_cast<bool>(out);
}

template <typename T>
bool ReadPod(std::istream& in, T* value) {
    in.read(reinterpret_cast<char*>(value), static_cast<std::streamsize>(sizeof(T)));
    return static_cast<bool>(in);
}

bool WriteStringBin(std::ostream& out, const std::string& value) {
    const std::uint64_t n = static_cast<std::uint64_t>(value.size());
    if (!WritePod(out, n)) {
        return false;
    }
    if (n > 0) {
        out.write(value.data(), static_cast<std::streamsize>(n));
    }
    return static_cast<bool>(out);
}

bool ReadStringBin(std::istream& in, std::string* value) {
    std::uint64_t n = 0;
    if (!ReadPod(in, &n)) {
        return false;
    }
    value->assign(static_cast<std::size_t>(n), '\0');
    if (n > 0) {
        in.read(value->data(), static_cast<std::streamsize>(n));
    }
    return static_cast<bool>(in);
}

bool WriteFloatMatrixBin(std::ostream& out, const ColMajorMatrix<float>& matrix) {
    const std::int32_t rows = matrix.rows;
    const std::int32_t cols = matrix.cols;
    const std::uint64_t count = static_cast<std::uint64_t>(matrix.data.size());
    if (!WritePod(out, rows) || !WritePod(out, cols) || !WritePod(out, count)) {
        return false;
    }
    if (count > 0) {
        out.write(reinterpret_cast<const char*>(matrix.data.data()),
                  static_cast<std::streamsize>(count * sizeof(float)));
    }
    return static_cast<bool>(out);
}

bool WriteCodeMatrixBin(std::ostream& out, const ColMajorMatrix<FullCode>& matrix) {
    const std::int32_t rows = matrix.rows;
    const std::int32_t cols = matrix.cols;
    const std::uint64_t count = static_cast<std::uint64_t>(matrix.data.size());
    if (!WritePod(out, rows) || !WritePod(out, cols) || !WritePod(out, count)) {
        return false;
    }
    if (count > 0) {
        out.write(reinterpret_cast<const char*>(matrix.data.data()),
                  static_cast<std::streamsize>(count * sizeof(FullCode)));
    }
    return static_cast<bool>(out);
}

bool ReadFloatMatrixBin(std::istream& in, ColMajorMatrix<float>* matrix) {
    std::int32_t rows = 0;
    std::int32_t cols = 0;
    std::uint64_t count = 0;
    if (!ReadPod(in, &rows) || !ReadPod(in, &cols) || !ReadPod(in, &count)) {
        return false;
    }
    if (rows < 0 || cols < 0 || count != static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(cols)) {
        return false;
    }
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->data.resize(static_cast<std::size_t>(count));
    if (count > 0) {
        in.read(reinterpret_cast<char*>(matrix->data.data()), static_cast<std::streamsize>(count * sizeof(float)));
    }
    return static_cast<bool>(in);
}

bool ReadCodeMatrixBin(std::istream& in, ColMajorMatrix<FullCode>* matrix) {
    std::int32_t rows = 0;
    std::int32_t cols = 0;
    std::uint64_t count = 0;
    if (!ReadPod(in, &rows) || !ReadPod(in, &cols) || !ReadPod(in, &count)) {
        return false;
    }
    if (rows < 0 || cols < 0 || count != static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(cols)) {
        return false;
    }
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->data.resize(static_cast<std::size_t>(count));
    if (count > 0) {
        in.read(reinterpret_cast<char*>(matrix->data.data()),
                static_cast<std::streamsize>(count * sizeof(FullCode)));
    }
    return static_cast<bool>(in);
}

bool WriteIntVectorBin(std::ostream& out, const std::vector<int>& vec) {
    const std::uint64_t n = static_cast<std::uint64_t>(vec.size());
    if (!WritePod(out, n)) {
        return false;
    }
    for (int v : vec) {
        const std::int32_t x = static_cast<std::int32_t>(v);
        if (!WritePod(out, x)) {
            return false;
        }
    }
    return static_cast<bool>(out);
}

bool ReadIntVectorBin(std::istream& in, std::vector<int>* vec) {
    std::uint64_t n = 0;
    if (!ReadPod(in, &n)) {
        return false;
    }
    vec->resize(static_cast<std::size_t>(n));
    for (std::uint64_t i = 0; i < n; ++i) {
        std::int32_t x = 0;
        if (!ReadPod(in, &x)) {
            return false;
        }
        (*vec)[static_cast<std::size_t>(i)] = static_cast<int>(x);
    }
    return true;
}

bool WriteVirtualEncodingBin(std::ostream& out, const VirtualEncoding* virt) {
    const std::uint8_t has_virt = (virt && virt->n_virtual() > 0) ? 1 : 0;
    if (!WritePod(out, has_virt)) {
        return false;
    }
    if (!has_virt) {
        return true;
    }
    const std::int32_t m = virt->m;
    const std::int32_t nlist = virt->nlist;
    if (!WritePod(out, m) || !WritePod(out, nlist)) {
        return false;
    }
    for (int cid = 0; cid < nlist; ++cid) {
        if (!WriteCodeMatrixBin(out, virt->B_by_cluster[static_cast<std::size_t>(cid)]) ||
            !WriteFloatMatrixBin(out, virt->a_by_cluster[static_cast<std::size_t>(cid)])) {
            return false;
        }
    }
    return true;
}

bool ReadVirtualEncodingBin(std::istream& in, VirtualEncoding* virt) {
    std::uint8_t has_virt = 0;
    if (!ReadPod(in, &has_virt)) {
        return false;
    }
    if (!virt) {
        if (!has_virt) {
            return true;
        }
        std::int32_t m = 0;
        std::int32_t nlist = 0;
        if (!ReadPod(in, &m) || !ReadPod(in, &nlist) || m < 0 || nlist < 0) {
            return false;
        }
        for (int cid = 0; cid < nlist; ++cid) {
            ColMajorMatrix<FullCode> B;
            ColMajorMatrix<float> a;
            if (!ReadCodeMatrixBin(in, &B) || !ReadFloatMatrixBin(in, &a)) {
                return false;
            }
        }
        return true;
    }
    *virt = {};
    if (!has_virt) {
        return true;
    }
    std::int32_t m = 0;
    std::int32_t nlist = 0;
    if (!ReadPod(in, &m) || !ReadPod(in, &nlist) || m < 0 || nlist < 0) {
        return false;
    }
    virt->m = m;
    virt->nlist = nlist;
    virt->B_by_cluster.resize(static_cast<std::size_t>(nlist));
    virt->a_by_cluster.resize(static_cast<std::size_t>(nlist));
    for (int cid = 0; cid < nlist; ++cid) {
        if (!ReadCodeMatrixBin(in, &virt->B_by_cluster[static_cast<std::size_t>(cid)]) ||
            !ReadFloatMatrixBin(in, &virt->a_by_cluster[static_cast<std::size_t>(cid)])) {
            return false;
        }
    }
    return true;
}

std::string TrainResumeSigHex(const Config& config, std::uint64_t* sig_out) {
    const std::uint64_t sig = ComputeTrainResumeSigU64(config);
    if (sig_out) {
        *sig_out = sig;
    }
    std::ostringstream oss;
    oss << "0x" << std::hex << std::setw(16) << std::setfill('0') << sig;
    return oss.str();
}

void WriteTrainCheckpointSidecars(const std::string& path,
                                  const Config& config,
                                  const TrainResult& result,
                                  int checkpoint_stage,
                                  const std::string& exp_root_hint,
                                  const char* storage_format,
                                  const char* caller) {
    const std::filesystem::path p(path);
    const std::string info_path = (p.string() + ".checkpoint.txt");
    const std::string cfg_path = (p.string() + ".config_snapshot.txt");

    std::ofstream info(info_path, std::ios::trunc);
    if (info.is_open()) {
        info << "# Train checkpoint metadata (sidecar)\n";
        info << "save_time_local = \"" << NowLocalISO8601() << "\"\n";
        info << "result_path = \"" << path << "\"\n";
        info << "storage_format = \"" << storage_format << "\"\n";
        if (checkpoint_stage == 1) {
            info << "checkpoint_stage = \"pre_init_linkage\" (C_root+R only)\n";
            info << "exp_root_hint = \"" << exp_root_hint << "\"\n";
        } else {
            info << "checkpoint_stage = \"full\"\n";
        }
        info << "io.config_file = \"" << config.io.config_file << "\"\n";
        info << "io.train_file = \"" << config.io.train_file << "\"\n";
        info << "io.load_date = \"" << config.io.load_date << "\"\n";
        info << "io.load_seq = \"" << config.io.load_seq << "\"\n";
        info << "train.init_enabled = " << (config.train.init_enabled ? "true" : "false") << "\n";
        info << "train.ckpt.enabled = " << (config.train.ckpt.enabled ? "true" : "false") << "\n";
        info << "train.ckpt.every_R = " << config.train.ckpt.every_R << "\n";
        info << "train.ckpt_after_init_basic = " << (config.train.ckpt_after_init_basic ? "true" : "false") << "\n";
        info << "train.exit_after_ckpt_init_basic = " << (config.train.exit_after_ckpt_init_basic ? "true" : "false") << "\n";
        info << "train.max_R_iters = " << config.train.max_R_iters << "\n";
        info << "result.R_iters = " << result.R_iters << "\n";
        info << "dataset.name = \"" << config.dataset.name << "\"\n";
        info << "dataset.ntrain = " << config.dataset.ntrain << "\n";
        info << "model.m = " << config.model.m << "\n";
        info << "model.h_vec = [";
        for (std::size_t i = 0; i < config.model.h_vec.size(); ++i) {
            if (i) info << ", ";
            info << config.model.h_vec[i];
        }
        info << "]\n";
        info << "\n# Notes:\n";
        info << "# - result.R_iters is the completed OPQ/global-R iteration count stored in the checkpoint.\n";
        info << "# - train.max_R_iters may be interpreted as \"additional iters\" when resuming (train.init_enabled=false).\n";
        info << "# - full config snapshot is embedded in the checkpoint and mirrored in the sidecar below.\n";
    } else {
        LogWarn(std::string(caller) + ": failed to write sidecar: " + info_path);
    }

    std::string cfg_err;
    if (!SaveConfigSnapshot(config, cfg_path, &cfg_err)) {
        LogWarn(std::string(caller) + ": failed to write config snapshot sidecar: " + cfg_path + " (" + cfg_err + ")");
    }
}

bool SaveTrainResultsBinImpl(const std::string& path,
                             const Config& config,
                             const TrainResult& result,
                             int checkpoint_stage,
                             const std::string& exp_root_hint,
                             std::string* error) {
    if (!EnsureParentDir(path, error)) {
        return false;
    }
    const int m = static_cast<int>(result.C_root.books.size());
    if (m <= 0) {
        if (error) {
            *error = "SaveTrainResultsBin: empty C_root.";
        }
        return false;
    }
    if (checkpoint_stage == 0 && static_cast<int>(result.C_one.books.size()) != m) {
        if (error) {
            *error = "SaveTrainResultsBin: C_root/C_one book count mismatch.";
        }
        return false;
    }
    if (checkpoint_stage == 1 && !result.C_one.books.empty()) {
        if (error) {
            *error = "SaveTrainResultsBin: expected empty C_one for pre-init-linkage checkpoint.";
        }
        return false;
    }
    if (result.R.rows <= 0 || result.R.cols <= 0) {
        if (error) {
            *error = "SaveTrainResultsBin: missing R.";
        }
        return false;
    }

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        if (error) {
            *error = "SaveTrainResultsBin: failed to open " + path;
        }
        return false;
    }

    const char magic[8] = {'S', 'T', 'L', 'Q', 'T', 'R', 'N', '1'};
    const std::uint32_t version = 1;
    const std::int32_t stage = checkpoint_stage;
    const std::int32_t book_count = m;
    std::uint64_t sig = 0;
    const std::string sig_hex = TrainResumeSigHex(config, &sig);
    const std::string snapshot = DumpConfigMini(config);
    const std::string exp_hint = checkpoint_stage == 1 ? exp_root_hint : result.exp_root_hint;
    out.write(magic, sizeof(magic));
    if (!WritePod(out, version) ||
        !WritePod(out, stage) ||
        !WritePod(out, book_count) ||
        !WriteStringBin(out, snapshot) ||
        !WriteStringBin(out, sig_hex) ||
        !WritePod(out, sig) ||
        !WriteStringBin(out, exp_hint) ||
        !WritePod(out, result.init_mse) ||
        !WritePod(out, result.beam_mse) ||
        !WritePod(out, result.icm_mse) ||
        !WritePod(out, result.linkage_mse_mean) ||
        !WritePod(out, result.linkage_ratio) ||
        !WritePod(out, result.linkage_max_depth) ||
        !WritePod(out, result.linkage_mean_depth) ||
        !WritePod(out, result.R_iters) ||
        !WritePod(out, result.final_linkaged_mse_rot)) {
        if (error) {
            *error = "SaveTrainResultsBin: failed to write metadata.";
        }
        return false;
    }
    for (const auto& book : result.C_root.books) {
        if (!WriteFloatMatrixBin(out, book)) {
            if (error) {
                *error = "SaveTrainResultsBin: failed to write C_root.";
            }
            return false;
        }
    }
    const std::int32_t c_one_count = checkpoint_stage == 0 ? m : 0;
    if (!WritePod(out, c_one_count)) {
        if (error) {
            *error = "SaveTrainResultsBin: failed to write C_one count.";
        }
        return false;
    }
    for (const auto& book : result.C_one.books) {
        if (!WriteFloatMatrixBin(out, book)) {
            if (error) {
                *error = "SaveTrainResultsBin: failed to write C_one.";
            }
            return false;
        }
    }
    if (!WriteFloatMatrixBin(out, result.R)) {
        if (error) {
            *error = "SaveTrainResultsBin: failed to write R.";
        }
        return false;
    }
    const std::uint64_t bad_count = static_cast<std::uint64_t>(result.is_bad_cluster.size());
    if (!WritePod(out, bad_count)) {
        if (error) {
            *error = "SaveTrainResultsBin: failed to write is_bad_cluster count.";
        }
        return false;
    }
    if (bad_count > 0) {
        const std::vector<std::uint8_t> bad = BoolToU8(result.is_bad_cluster);
        out.write(reinterpret_cast<const char*>(bad.data()), static_cast<std::streamsize>(bad.size()));
    }
    if (!out.good()) {
        if (error) {
            *error = "SaveTrainResultsBin: failed while writing " + path;
        }
        return false;
    }

    WriteTrainCheckpointSidecars(path, config, result, checkpoint_stage, exp_root_hint, "bin", "SaveTrainResultsBin");
    return true;
}

bool LoadTrainResultsBinImpl(const std::string& path, int m, TrainResult* result, std::string* error) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        if (error) {
            *error = "LoadTrainResultsBin: failed to open " + path;
        }
        return false;
    }
    char magic[8] = {};
    in.read(magic, sizeof(magic));
    const char expected[8] = {'S', 'T', 'L', 'Q', 'T', 'R', 'N', '1'};
    if (!in || std::memcmp(magic, expected, sizeof(expected)) != 0) {
        if (error) {
            *error = "LoadTrainResultsBin: invalid STLQ train-result magic.";
        }
        return false;
    }
    std::uint32_t version = 0;
    std::int32_t checkpoint_stage = 0;
    std::int32_t book_count = 0;
    std::string snapshot;
    std::string sig_hex;
    std::uint64_t sig = 0;
    std::string exp_root_hint;
    if (!ReadPod(in, &version) || version != 1 ||
        !ReadPod(in, &checkpoint_stage) ||
        !ReadPod(in, &book_count) ||
        !ReadStringBin(in, &snapshot) ||
        !ReadStringBin(in, &sig_hex) ||
        !ReadPod(in, &sig) ||
        !ReadStringBin(in, &exp_root_hint) ||
        !ReadPod(in, &result->init_mse) ||
        !ReadPod(in, &result->beam_mse) ||
        !ReadPod(in, &result->icm_mse) ||
        !ReadPod(in, &result->linkage_mse_mean) ||
        !ReadPod(in, &result->linkage_ratio) ||
        !ReadPod(in, &result->linkage_max_depth) ||
        !ReadPod(in, &result->linkage_mean_depth) ||
        !ReadPod(in, &result->R_iters) ||
        !ReadPod(in, &result->final_linkaged_mse_rot)) {
        if (error) {
            *error = "LoadTrainResultsBin: failed to read metadata.";
        }
        return false;
    }
    if (m <= 0 || book_count != m || (checkpoint_stage != 0 && checkpoint_stage != 1)) {
        if (error) {
            *error = "LoadTrainResultsBin: invalid m or checkpoint stage.";
        }
        return false;
    }

    std::vector<ColMajorMatrix<float>> C_root(static_cast<std::size_t>(m));
    for (int i = 0; i < m; ++i) {
        if (!ReadFloatMatrixBin(in, &C_root[static_cast<std::size_t>(i)])) {
            if (error) {
                *error = "LoadTrainResultsBin: failed to read C_root.";
            }
            return false;
        }
    }
    std::int32_t c_one_count = 0;
    if (!ReadPod(in, &c_one_count) || c_one_count < 0) {
        if (error) {
            *error = "LoadTrainResultsBin: failed to read C_one count.";
        }
        return false;
    }
    std::vector<ColMajorMatrix<float>> C_one(static_cast<std::size_t>(c_one_count));
    for (int i = 0; i < c_one_count; ++i) {
        if (!ReadFloatMatrixBin(in, &C_one[static_cast<std::size_t>(i)])) {
            if (error) {
                *error = "LoadTrainResultsBin: failed to read C_one.";
            }
            return false;
        }
    }
    ColMajorMatrix<float> R;
    if (!ReadFloatMatrixBin(in, &R)) {
        if (error) {
            *error = "LoadTrainResultsBin: failed to read R.";
        }
        return false;
    }
    std::uint64_t bad_count = 0;
    if (!ReadPod(in, &bad_count)) {
        if (error) {
            *error = "LoadTrainResultsBin: failed to read is_bad_cluster count.";
        }
        return false;
    }
    std::vector<std::uint8_t> bad(static_cast<std::size_t>(bad_count));
    if (bad_count > 0) {
        in.read(reinterpret_cast<char*>(bad.data()), static_cast<std::streamsize>(bad.size()));
        if (!in) {
            if (error) {
                *error = "LoadTrainResultsBin: failed to read is_bad_cluster.";
            }
            return false;
        }
    }

    result->C_root.d = C_root.empty() ? 0 : C_root.front().rows;
    result->C_root.books = std::move(C_root);
    result->C_root.h_vec.clear();
    for (const auto& book : result->C_root.books) {
        result->C_root.h_vec.push_back(book.cols);
    }
    if (checkpoint_stage == 0) {
        if (c_one_count != m) {
            if (error) {
                *error = "LoadTrainResultsBin: full checkpoint missing C_one books.";
            }
            return false;
        }
        result->C_one.d = C_one.empty() ? 0 : C_one.front().rows;
        result->C_one.books = std::move(C_one);
        result->C_one.h_vec.clear();
        for (const auto& book : result->C_one.books) {
            result->C_one.h_vec.push_back(book.cols);
        }
    } else {
        result->C_one = {};
        result->C_one.d = result->C_root.d;
    }
    result->R = std::move(R);
    result->is_bad_cluster = U8ToBool(bad);
    result->train_linkage = {};
    result->resume_sig_u64 = sig;
    result->exp_root_hint = std::move(exp_root_hint);
    result->checkpoint_stage = checkpoint_stage;
    return true;
}

}  // namespace

std::string MakeUniqueDatedPath(const std::string& base_path, std::string* error) {
    if (base_path.empty()) {
        if (error) {
            *error = "MakeUniqueDatedPath: base_path is empty.";
        }
        return {};
    }
    std::filesystem::path p(base_path);
    const std::filesystem::path dir = p.parent_path();
    const std::string stem = p.stem().string();
    const std::string ext = p.extension().string();
    const std::string date = TodayYYYYMMDD();

    auto make_path = [&](int idx) -> std::filesystem::path {
        std::ostringstream name;
        name << stem << "_" << date;
        if (idx > 0) {
            name << "_" << std::setfill('0') << std::setw(3) << idx;
        }
        name << ext;
        return dir.empty() ? std::filesystem::path(name.str()) : (dir / name.str());
    };

    std::error_code ec;
    std::filesystem::path candidate = make_path(0);
    if (!std::filesystem::exists(candidate, ec) && !ec) {
        return candidate.string();
    }
    for (int idx = 1; idx <= 999; ++idx) {
        candidate = make_path(idx);
        ec.clear();
        if (!std::filesystem::exists(candidate, ec) && !ec) {
            return candidate.string();
        }
    }
    if (error) {
        *error = "MakeUniqueDatedPath: failed to find a unique filename for " + base_path;
    }
    return {};
}

std::string MakeDatedPathForLoad(const std::string& base_path,
                                 const std::string& load_date,
                                 const std::string& load_seq,
                                 std::string* error) {
    if (base_path.empty()) {
        if (error) {
            *error = "MakeDatedPathForLoad: base_path is empty.";
        }
        return {};
    }

    // Compatibility mode: load the path exactly as provided (no date/seq suffixing).
    // Useful when loading existing Julia-produced files like:
    //   results/siftsmall/op2brdc_ex_m5.h5
    // which do not follow the C++ dated naming convention.
    if (load_date == "raw") {
        std::filesystem::path out(base_path);
        std::error_code ec;
        if (!std::filesystem::exists(out, ec) || ec) {
            if (error) {
                *error = "MakeDatedPathForLoad: raw file not found: " + out.string();
            }
            return {};
        }
        return out.string();
    }

    const std::string date = load_date.empty() ? TodayYYYYMMDD() : load_date;
    if (date.size() != 8 ||
        !std::all_of(date.begin(), date.end(), [](unsigned char c) { return c >= '0' && c <= '9'; })) {
        if (error) {
            *error = "MakeDatedPathForLoad: io.load_date must be YYYYMMDD digits (got: " + date + ")";
        }
        return {};
    }

    std::string seq = load_seq;
    if (!seq.empty()) {
        if (!std::all_of(seq.begin(), seq.end(), [](unsigned char c) { return c >= '0' && c <= '9'; })) {
            if (error) {
                *error = "MakeDatedPathForLoad: io.load_seq must be digits (got: " + seq + ")";
            }
            return {};
        }
        int seq_val = 0;
        try {
            seq_val = std::stoi(seq);
        } catch (...) {
            if (error) {
                *error = "MakeDatedPathForLoad: failed to parse io.load_seq: " + seq;
            }
            return {};
        }
        if (seq_val < 0 || seq_val > 999) {
            if (error) {
                *error = "MakeDatedPathForLoad: io.load_seq must be in [0,999] (got: " + seq + ")";
            }
            return {};
        }
        std::ostringstream oss;
        oss << std::setfill('0') << std::setw(3) << seq_val;
        seq = oss.str();
    }

    std::filesystem::path p(base_path);
    const std::filesystem::path dir = p.parent_path();
    const std::string stem = p.stem().string();
    const std::string ext = p.extension().string();

    std::ostringstream name;
    name << stem << "_" << date;
    if (!seq.empty()) {
        name << "_" << seq;
    }
    name << ext;

    std::filesystem::path out = dir.empty() ? std::filesystem::path(name.str()) : (dir / name.str());
    std::error_code ec;
    if (!std::filesystem::exists(out, ec) || ec) {
        if (error) {
            *error = "MakeDatedPathForLoad: file not found: " + out.string();
        }
        return {};
    }
    return out.string();
}

int ComputeMaxDepthFromParent(const std::vector<int>& parent) {
    const int n = static_cast<int>(parent.size());
    if (n <= 0) {
        return 0;
    }
    std::vector<int> depth(static_cast<std::size_t>(n), -1);
    int best = 0;
    for (int i = 0; i < n; ++i) {
        int cur = i;
        int d = 0;
        while (cur >= 0 && cur < n) {
            int known = depth[static_cast<std::size_t>(cur)];
            if (known >= 0) {
                d += known;
                break;
            }
            int p = parent[static_cast<std::size_t>(cur)];
            if (p == 0) {
                break;
            }
            ++d;
            cur = p - 1;
        }
        // Backfill along the linkage for amortized O(n).
        cur = i;
        int rem = d;
        while (cur >= 0 && cur < n) {
            if (depth[static_cast<std::size_t>(cur)] >= 0) {
                break;
            }
            depth[static_cast<std::size_t>(cur)] = rem;
            if (rem == 0) {
                break;
            }
            int p = parent[static_cast<std::size_t>(cur)];
            if (p == 0) {
                break;
            }
            --rem;
            cur = p - 1;
        }
        best = std::max(best, d);
    }
    return best;
}

bool SaveTrainResults(const std::string& path,
                      const Config& config,
                      const TrainResult& result,
                      std::string* error) {
    if (ChooseTrainResultSaveFormat(config, path) == TrainResultStorageFormat::kBin) {
        return SaveTrainResultsBinImpl(path, config, result, 0, "", error);
    }
#if !(defined(STLQ_ENABLE_HDF5) && STLQ_ENABLE_HDF5)
    if (error) {
        *error = "SaveTrainResults: HDF5 support is disabled; set io.result_format=\"bin\" or use a .stlqbin train_file.";
    }
    return false;
#else
    if (!EnsureParentDir(path, error)) {
        return false;
    }
    Hdf5Writer writer(path);
    if (!writer.IsOpen()) {
        if (error) {
            *error = "Failed to open train output file: " + path;
        }
        return false;
    }

    const int m = static_cast<int>(result.C_root.books.size());
    if (m == 0 || static_cast<int>(result.C_one.books.size()) != m) {
        if (error) {
            *error = "SaveTrainResults: C_root/C_one book count mismatch.";
        }
        return false;
    }

    if (!writer.WriteString("meta/config_snapshot", DumpConfigMini(config))) {
        if (error) {
            *error = "SaveTrainResults: failed to write meta/config_snapshot.";
        }
        return false;
    }
    {
        if (!writer.WriteString("meta/train/resume_sig_hex", TrainResumeSigHex(config, nullptr))) {
            if (error) {
                *error = "SaveTrainResults: failed to write meta/train/resume_sig_hex.";
            }
            return false;
        }
    }
    // Checkpoint stage marker for resume (0 => normal/full checkpoint).
    writer.WriteScalar("meta/train/checkpoint_stage", 0);
    if (result.init_mse >= 0.0f) {
        writer.WriteScalar("meta/train/init_mse", result.init_mse);
    }
    if (result.beam_mse >= 0.0f) {
        writer.WriteScalar("meta/train/beam_mse", result.beam_mse);
    }
    if (result.icm_mse >= 0.0f) {
        writer.WriteScalar("meta/train/icm_mse", result.icm_mse);
    }
    if (result.linkage_mse_mean >= 0.0f) {
        writer.WriteScalar("meta/train/linkage_mse_mean", result.linkage_mse_mean);
    }
    if (result.linkage_ratio >= 0.0f) {
        writer.WriteScalar("meta/train/linkage_ratio", result.linkage_ratio);
    }
    if (result.linkage_max_depth >= 0) {
        writer.WriteScalar("meta/train/linkage_max_depth", result.linkage_max_depth);
    }
    if (result.linkage_mean_depth >= 0.0f) {
        writer.WriteScalar("meta/train/linkage_mean_depth", result.linkage_mean_depth);
    }
    if (result.R_iters >= 0) {
        writer.WriteScalar("meta/train/R_iters", result.R_iters);
    }
    if (result.final_linkaged_mse_rot >= 0.0f) {
        writer.WriteScalar("meta/train/final_linkaged_mse_rot", result.final_linkaged_mse_rot);
    }

    // Julia compatibility: save_c_R writes C_1..C_(2m) and R.
    for (int i = 0; i < m; ++i) {
        if (!writer.WriteMatrix("C_" + std::to_string(i + 1),
                                result.C_root.books[static_cast<std::size_t>(i)])) {
            if (error) {
                *error = "SaveTrainResults: failed to write C_root book C_" + std::to_string(i + 1);
            }
            return false;
        }
    }
    for (int i = 0; i < m; ++i) {
        if (!writer.WriteMatrix("C_" + std::to_string(m + i + 1),
                                result.C_one.books[static_cast<std::size_t>(i)])) {
            if (error) {
                *error = "SaveTrainResults: failed to write C_one book C_" + std::to_string(m + i + 1);
            }
            return false;
        }
    }
    if (!writer.WriteMatrix("R", result.R)) {
        if (error) {
            *error = "SaveTrainResults: failed to write R.";
        }
        return false;
    }

    if (!result.is_bad_cluster.empty()) {
        const std::vector<std::uint8_t> is_bad_u8 = BoolToU8(result.is_bad_cluster);
        if (!writer.WriteVector("is_bad_cluster", is_bad_u8)) {
            if (error) {
                *error = "SaveTrainResults: failed to write is_bad_cluster.";
            }
            return false;
        }
    }

    WriteTrainCheckpointSidecars(path, config, result, 0, "", "hdf5", "SaveTrainResults");
    return true;
#endif
}

bool SaveTrainResultsPreInitLinkage(const std::string& path,
                                 const Config& config,
                                 const TrainResult& result,
                                 const std::string& exp_root_hint,
                                 std::string* error) {
    if (ChooseTrainResultSaveFormat(config, path) == TrainResultStorageFormat::kBin) {
        return SaveTrainResultsBinImpl(path, config, result, 1, exp_root_hint, error);
    }
#if !(defined(STLQ_ENABLE_HDF5) && STLQ_ENABLE_HDF5)
    if (error) {
        *error = "SaveTrainResultsPreInitLinkage: HDF5 support is disabled; set io.result_format=\"bin\" or use a .stlqbin train_file.";
    }
    return false;
#else
    if (!EnsureParentDir(path, error)) {
        return false;
    }
    Hdf5Writer writer(path);
    if (!writer.IsOpen()) {
        if (error) {
            *error = "Failed to open train output file: " + path;
        }
        return false;
    }

    const int m = static_cast<int>(result.C_root.books.size());
    if (m <= 0) {
        if (error) {
            *error = "SaveTrainResultsPreInitLinkage: empty C_root.";
        }
        return false;
    }
    if (static_cast<int>(result.C_one.books.size()) != 0) {
        if (error) {
            *error = "SaveTrainResultsPreInitLinkage: expected empty C_one (pre-init-linkage checkpoint).";
        }
        return false;
    }
    if (result.R.rows <= 0 || result.R.cols <= 0) {
        if (error) {
            *error = "SaveTrainResultsPreInitLinkage: missing R.";
        }
        return false;
    }

    if (!writer.WriteString("meta/config_snapshot", DumpConfigMini(config))) {
        if (error) {
            *error = "SaveTrainResultsPreInitLinkage: failed to write meta/config_snapshot.";
        }
        return false;
    }
    {
        if (!writer.WriteString("meta/train/resume_sig_hex", TrainResumeSigHex(config, nullptr))) {
            if (error) {
                *error = "SaveTrainResultsPreInitLinkage: failed to write meta/train/resume_sig_hex.";
            }
            return false;
        }
    }
    writer.WriteScalar("meta/train/checkpoint_stage", 1);
    if (!exp_root_hint.empty()) {
        writer.WriteString("meta/train/exp_root_hint", exp_root_hint);
    }
    if (result.R_iters >= 0) {
        writer.WriteScalar("meta/train/R_iters", result.R_iters);
    }

    // Save only C_root (C_1..C_m) and R. C_one is intentionally absent.
    for (int i = 0; i < m; ++i) {
        if (!writer.WriteMatrix("C_" + std::to_string(i + 1),
                                result.C_root.books[static_cast<std::size_t>(i)])) {
            if (error) {
                *error = "SaveTrainResultsPreInitLinkage: failed to write C_root book C_" + std::to_string(i + 1);
            }
            return false;
        }
    }
    if (!writer.WriteMatrix("R", result.R)) {
        if (error) {
            *error = "SaveTrainResultsPreInitLinkage: failed to write R.";
        }
        return false;
    }

    WriteTrainCheckpointSidecars(path, config, result, 1, exp_root_hint, "hdf5", "SaveTrainResultsPreInitLinkage");
    return true;
#endif
}

bool LoadTrainResults(const std::string& base_path,
                      const std::string& load_date,
                      const std::string& load_seq,
                      int m,
                      TrainResult* result,
                      std::string* error) {
    if (!result) {
        if (error) {
            *error = "LoadTrainResults: TrainResult is null.";
        }
        return false;
    }
    const std::string path = MakeDatedPathForLoad(base_path, load_date, load_seq, error);
    if (path.empty()) {
        return false;
    }
    LogInfo("Loading train results from: " + path);

    if (ChooseTrainResultLoadFormat(path) == TrainResultStorageFormat::kBin) {
        return LoadTrainResultsBinImpl(path, m, result, error);
    }
#if !(defined(STLQ_ENABLE_HDF5) && STLQ_ENABLE_HDF5)
    if (error) {
        *error = "LoadTrainResults: HDF5 support is disabled; use a .stlqbin train checkpoint or rebuild with STLQ_ENABLE_HDF5=ON.";
    }
    return false;
#else
    Hdf5Reader reader(path);
    if (!reader.IsOpen()) {
        if (error) {
            *error = "LoadTrainResults: failed to open " + path;
        }
        return false;
    }

    if (m <= 0) {
        if (error) {
            *error = "LoadTrainResults: invalid m.";
        }
        return false;
    }

    int checkpoint_stage = 0;
    if (reader.Has("meta/train/checkpoint_stage")) {
        reader.ReadScalar("meta/train/checkpoint_stage", &checkpoint_stage);
    }

    std::vector<ColMajorMatrix<float>> C_root;
    std::vector<ColMajorMatrix<float>> C_one;
    C_root.resize(static_cast<std::size_t>(m));
    if (checkpoint_stage == 0) {
        C_one.resize(static_cast<std::size_t>(m));
    }
    for (int i = 0; i < m; ++i) {
        if (!reader.ReadMatrix("C_" + std::to_string(i + 1), &C_root[static_cast<std::size_t>(i)])) {
            if (error) {
                *error = "LoadTrainResults: missing C_" + std::to_string(i + 1);
            }
            return false;
        }
    }
    if (checkpoint_stage == 0) {
        for (int i = 0; i < m; ++i) {
            if (!reader.ReadMatrix("C_" + std::to_string(m + i + 1), &C_one[static_cast<std::size_t>(i)])) {
                if (error) {
                    *error = "LoadTrainResults: missing C_" + std::to_string(m + i + 1);
                }
                return false;
            }
        }
    }

    ColMajorMatrix<float> R;
    if (!reader.ReadMatrix("R", &R)) {
        if (error) {
            *error = "LoadTrainResults: missing R";
        }
        return false;
    }

    result->C_root.d = C_root.front().rows;
    result->C_root.books = std::move(C_root);
    result->C_root.h_vec.clear();
    for (const auto& book : result->C_root.books) {
        result->C_root.h_vec.push_back(book.cols);
    }

    if (checkpoint_stage == 0) {
        result->C_one.d = C_one.front().rows;
        result->C_one.books = std::move(C_one);
        result->C_one.h_vec.clear();
        for (const auto& book : result->C_one.books) {
            result->C_one.h_vec.push_back(book.cols);
        }
    } else {
        result->C_one = {};
        // Keep d consistent for downstream checks; C_one books will be trained after resuming init-linkage.
        result->C_one.d = result->C_root.d;
    }

    result->R = std::move(R);
    result->is_bad_cluster.clear();
    if (reader.Has("is_bad_cluster")) {
        std::vector<std::uint8_t> tmp;
        reader.ReadVector("is_bad_cluster", &tmp);
        result->is_bad_cluster = U8ToBool(tmp);
    }
    result->train_linkage = {};

    // Optional meta: R_iters (for resume training).
    result->R_iters = 0;
    if (reader.Has("meta/train/R_iters")) {
        int r_iters = 0;
        if (reader.ReadScalar("meta/train/R_iters", &r_iters)) {
            result->R_iters = std::max(0, r_iters);
        }
    }
    result->resume_sig_u64 = 0;
    if (reader.Has("meta/train/resume_sig_hex")) {
        std::string s;
        if (reader.ReadString("meta/train/resume_sig_hex", &s)) {
            const char* p = s.c_str();
            if (s.size() >= 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
                p += 2;
            }
            std::uint64_t sig = 0;
            std::stringstream ss{std::string(p)};
            ss >> std::hex >> sig;
            if (!ss.fail()) {
                result->resume_sig_u64 = sig;
            }
        }
    }
    result->exp_root_hint.clear();
    if (reader.Has("meta/train/exp_root_hint")) {
        std::string s;
        if (reader.ReadString("meta/train/exp_root_hint", &s)) {
            result->exp_root_hint = std::move(s);
        }
    }
    result->checkpoint_stage = checkpoint_stage;
    return true;
#endif
}

bool SaveBaseResultsHqBinImpl(const std::string& path,
                              const Config& config,
                              const BaseEncoding& base,
                              float base_error,
                              int n_base_real,
                              float beam_error,
                              const VirtualEncoding* virt,
                              std::string* error) {
    if (!EnsureParentDir(path, error)) {
        return false;
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        if (error) {
            *error = "SaveBaseResultsHqBin: failed to open " + path;
        }
        return false;
    }
    const char magic[8] = {'S', 'T', 'L', 'Q', 'B', 'A', 'S', '1'};
    const std::uint32_t version = 1;
    const std::string snapshot = DumpConfigMini(config);
    out.write(magic, sizeof(magic));
    if (!WritePod(out, version) ||
        !WriteStringBin(out, snapshot) ||
        !WritePod(out, base_error) ||
        !WritePod(out, n_base_real) ||
        !WritePod(out, beam_error) ||
        !WriteCodeMatrixBin(out, base.B) ||
        !WriteFloatMatrixBin(out, base.a) ||
        !WriteVirtualEncodingBin(out, virt)) {
        if (error) {
            *error = "SaveBaseResultsHqBin: failed while writing " + path;
        }
        return false;
    }
    return true;
}

bool LoadBaseResultsHqBinImpl(const std::string& path,
                              BaseEncoding* base,
                              int* n_base_real,
                              VirtualEncoding* virt,
                              std::string* error) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        if (error) {
            *error = "LoadBaseResultsHqBin: failed to open " + path;
        }
        return false;
    }
    char magic[8] = {};
    in.read(magic, sizeof(magic));
    const char expected[8] = {'S', 'T', 'L', 'Q', 'B', 'A', 'S', '1'};
    std::uint32_t version = 0;
    std::string snapshot;
    float base_error = -1.0f;
    int n_real = -1;
    float beam_error = -1.0f;
    if (!in || std::memcmp(magic, expected, sizeof(expected)) != 0 ||
        !ReadPod(in, &version) || version != 1 ||
        !ReadStringBin(in, &snapshot) ||
        !ReadPod(in, &base_error) ||
        !ReadPod(in, &n_real) ||
        !ReadPod(in, &beam_error) ||
        !ReadCodeMatrixBin(in, &base->B) ||
        !ReadFloatMatrixBin(in, &base->a) ||
        !ReadVirtualEncodingBin(in, virt)) {
        (void)base_error;
        (void)beam_error;
        if (error) {
            *error = "LoadBaseResultsHqBin: invalid or truncated file " + path;
        }
        return false;
    }
    if (n_base_real) {
        *n_base_real = n_real;
    }
    return true;
}

bool SaveBaseResultsSTLQBinImpl(const std::string& path,
                                const Config& config,
                                const BaseEncoding& base,
                                const std::vector<int>& parent,
                                const std::vector<int>* cluster_id,
                                float base_error,
                                int n_base_real,
                                float linkage_ratio,
                                int linkage_max_depth,
                                float linkage_mean_depth,
                                const std::vector<bool>* is_bad_cluster,
                                const VirtualEncoding* virt,
                                std::string* error) {
    if (!EnsureParentDir(path, error)) {
        return false;
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        if (error) {
            *error = "SaveBaseResultsSTLQBin: failed to open " + path;
        }
        return false;
    }
    const char magic[8] = {'S', 'T', 'L', 'Q', 'L', 'N', 'K', '1'};
    const std::uint32_t version = 1;
    const std::string snapshot = DumpConfigMini(config);
    const std::uint8_t has_cluster_id = (cluster_id && !cluster_id->empty()) ? 1 : 0;
    const std::uint8_t has_is_bad = (is_bad_cluster && !is_bad_cluster->empty()) ? 1 : 0;
    out.write(magic, sizeof(magic));
    if (!WritePod(out, version) ||
        !WriteStringBin(out, snapshot) ||
        !WritePod(out, base_error) ||
        !WritePod(out, n_base_real) ||
        !WritePod(out, linkage_ratio) ||
        !WritePod(out, linkage_max_depth) ||
        !WritePod(out, linkage_mean_depth) ||
        !WriteCodeMatrixBin(out, base.B) ||
        !WriteFloatMatrixBin(out, base.a) ||
        !WriteIntVectorBin(out, parent) ||
        !WritePod(out, has_cluster_id)) {
        if (error) {
            *error = "SaveBaseResultsSTLQBin: failed while writing header/body.";
        }
        return false;
    }
    if (has_cluster_id && !WriteIntVectorBin(out, *cluster_id)) {
        if (error) {
            *error = "SaveBaseResultsSTLQBin: failed while writing cluster_id.";
        }
        return false;
    }
    if (!WritePod(out, has_is_bad)) {
        if (error) {
            *error = "SaveBaseResultsSTLQBin: failed while writing is_bad marker.";
        }
        return false;
    }
    if (has_is_bad) {
        const std::vector<std::uint8_t> bad = BoolToU8(*is_bad_cluster);
        const std::uint64_t n = static_cast<std::uint64_t>(bad.size());
        if (!WritePod(out, n)) {
            if (error) {
                *error = "SaveBaseResultsSTLQBin: failed while writing is_bad size.";
            }
            return false;
        }
        out.write(reinterpret_cast<const char*>(bad.data()), static_cast<std::streamsize>(bad.size()));
    }
    if (!WriteVirtualEncodingBin(out, virt) || !out.good()) {
        if (error) {
            *error = "SaveBaseResultsSTLQBin: failed while writing " + path;
        }
        return false;
    }
    return true;
}

bool LoadBaseResultsSTLQBinImpl(const std::string& path,
                                BaseEncoding* base,
                                std::vector<int>* parent,
                                std::vector<int>* cluster_id,
                                int* n_base_real,
                                VirtualEncoding* virt,
                                std::string* error) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        if (error) {
            *error = "LoadBaseResultsSTLQBin: failed to open " + path;
        }
        return false;
    }
    char magic[8] = {};
    in.read(magic, sizeof(magic));
    const char expected[8] = {'S', 'T', 'L', 'Q', 'L', 'N', 'K', '1'};
    std::uint32_t version = 0;
    std::string snapshot;
    float base_error = -1.0f;
    int n_real = -1;
    float linkage_ratio = -1.0f;
    int linkage_max_depth = -1;
    float linkage_mean_depth = -1.0f;
    std::uint8_t has_cluster_id = 0;
    if (!in || std::memcmp(magic, expected, sizeof(expected)) != 0 ||
        !ReadPod(in, &version) || version != 1 ||
        !ReadStringBin(in, &snapshot) ||
        !ReadPod(in, &base_error) ||
        !ReadPod(in, &n_real) ||
        !ReadPod(in, &linkage_ratio) ||
        !ReadPod(in, &linkage_max_depth) ||
        !ReadPod(in, &linkage_mean_depth) ||
        !ReadCodeMatrixBin(in, &base->B) ||
        !ReadFloatMatrixBin(in, &base->a) ||
        !ReadIntVectorBin(in, parent) ||
        !ReadPod(in, &has_cluster_id)) {
        (void)base_error;
        (void)linkage_ratio;
        (void)linkage_max_depth;
        (void)linkage_mean_depth;
        if (error) {
            *error = "LoadBaseResultsSTLQBin: invalid or truncated file " + path;
        }
        return false;
    }
    if (cluster_id) {
        cluster_id->clear();
    }
    if (has_cluster_id) {
        std::vector<int> tmp;
        if (!ReadIntVectorBin(in, &tmp)) {
            if (error) {
                *error = "LoadBaseResultsSTLQBin: failed to read cluster_id.";
            }
            return false;
        }
        if (cluster_id) {
            *cluster_id = std::move(tmp);
        }
    }
    std::uint8_t has_is_bad = 0;
    if (!ReadPod(in, &has_is_bad)) {
        if (error) {
            *error = "LoadBaseResultsSTLQBin: failed to read is_bad marker.";
        }
        return false;
    }
    if (has_is_bad) {
        std::uint64_t n = 0;
        if (!ReadPod(in, &n)) {
            if (error) {
                *error = "LoadBaseResultsSTLQBin: failed to read is_bad size.";
            }
            return false;
        }
        std::vector<std::uint8_t> ignored(static_cast<std::size_t>(n));
        if (n > 0) {
            in.read(reinterpret_cast<char*>(ignored.data()), static_cast<std::streamsize>(ignored.size()));
        }
    }
    if (!ReadVirtualEncodingBin(in, virt)) {
        if (error) {
            *error = "LoadBaseResultsSTLQBin: failed to read virtual payload.";
        }
        return false;
    }
    if (n_base_real) {
        *n_base_real = n_real;
    }
    return true;
}

bool SaveBaseResultsHq(const std::string& path,
                       const Config& config,
                       const BaseEncoding& base,
                       float base_error,
                       int n_base_real,
                       float beam_error,
                       const VirtualEncoding* virt,
                       std::string* error) {
    if (ChooseResultSaveFormat(config, path) == ResultStorageFormat::kBin) {
        return SaveBaseResultsHqBinImpl(path, config, base, base_error, n_base_real, beam_error, virt, error);
    }
#if !(defined(STLQ_ENABLE_HDF5) && STLQ_ENABLE_HDF5)
    (void)path;
    (void)config;
    (void)base;
    (void)base_error;
    (void)n_base_real;
    (void)beam_error;
    (void)virt;
    if (error) {
        *error = "SaveBaseResultsHq: HDF5 support is disabled; non-large HDF5 base result IO is unavailable.";
    }
    return false;
#else
    if (!EnsureParentDir(path, error)) {
        return false;
    }
    Hdf5Writer writer(path);
    if (!writer.IsOpen()) {
        if (error) {
            *error = "Failed to open base output file: " + path;
        }
        return false;
    }
    if (!writer.WriteMatrix("B_base", base.B) ||
        !writer.WriteMatrix("a_base", base.a) ||
        !writer.WriteScalar("base_error", base_error)) {
        if (error) {
            *error = "SaveBaseResultsHq: failed to write datasets.";
        }
        return false;
    }
    if (virt && virt->n_virtual() > 0) {
        const int nlist = virt->nlist;
        std::vector<int> nvirt_by_cluster(static_cast<std::size_t>(nlist), 0);
        for (int cid = 0; cid < nlist; ++cid) {
            nvirt_by_cluster[static_cast<std::size_t>(cid)] = virt->n_virtual(cid);
        }
        if (!writer.WriteVector("virtual/nvirt_by_cluster", nvirt_by_cluster)) {
            if (error) {
                *error = "SaveBaseResultsHq: failed to write virtual datasets.";
            }
            return false;
        }
        for (int cid = 0; cid < nlist; ++cid) {
            const int nvc = nvirt_by_cluster[static_cast<std::size_t>(cid)];
            if (nvc <= 0) {
                continue;
            }
            const auto& Bc = virt->B_by_cluster[static_cast<std::size_t>(cid)];
            const auto& ac = virt->a_by_cluster[static_cast<std::size_t>(cid)];
            if (!writer.WriteMatrix("virtual/B_" + std::to_string(cid), Bc) ||
                !writer.WriteMatrix("virtual/a_" + std::to_string(cid), ac)) {
                if (error) {
                    *error = "SaveBaseResultsHq: failed to write per-cluster virtual datasets.";
                }
                return false;
            }
        }
        writer.WriteScalar("meta/n_base_virtual", virt->n_virtual());
        writer.WriteScalar("meta/virtual/nlist", virt->nlist);
    }
    writer.WriteString("meta/config_snapshot", DumpConfigMini(config));
    if (beam_error >= 0.0f) {
        writer.WriteScalar("meta/base/beam_mse", beam_error);
    }
    if (base_error >= 0.0f) {
        writer.WriteScalar("meta/base/final_mse", base_error);
    }
    if (n_base_real >= 0) {
        writer.WriteScalar("meta/n_base_real", n_base_real);
    }
    return true;
#endif
}

bool SaveBaseResultsSTLQ(const std::string& path,
                           const Config& config,
                           const BaseEncoding& base,
                           const std::vector<int>& parent,
                           const std::vector<int>* cluster_id,
                           float base_error,
                           int n_base_real,
                           float linkage_ratio,
                           int linkage_max_depth,
                           float linkage_mean_depth,
                           const std::vector<bool>* is_bad_cluster,
                           const VirtualEncoding* virt,
                           std::string* error) {
    if (ChooseResultSaveFormat(config, path) == ResultStorageFormat::kBin) {
        return SaveBaseResultsSTLQBinImpl(path, config, base, parent, cluster_id, base_error, n_base_real,
                                          linkage_ratio, linkage_max_depth, linkage_mean_depth,
                                          is_bad_cluster, virt, error);
    }
#if !(defined(STLQ_ENABLE_HDF5) && STLQ_ENABLE_HDF5)
    (void)path;
    (void)config;
    (void)base;
    (void)parent;
    (void)cluster_id;
    (void)base_error;
    (void)n_base_real;
    (void)linkage_ratio;
    (void)linkage_max_depth;
    (void)linkage_mean_depth;
    (void)is_bad_cluster;
    (void)virt;
    if (error) {
        *error = "SaveBaseResultsSTLQ: HDF5 support is disabled; non-large HDF5 linkage result IO is unavailable.";
    }
    return false;
#else
    if (!EnsureParentDir(path, error)) {
        return false;
    }
    Hdf5Writer writer(path);
    if (!writer.IsOpen()) {
        if (error) {
            *error = "Failed to open linkage output file: " + path;
        }
        return false;
    }
    writer.WriteString("meta/index_dtype", config.io.index_dtype);
    if (!writer.WriteMatrix("B_base", base.B) ||
        !writer.WriteMatrix("a_base", base.a) ||
        !(config.io.index_dtype == "uint"
              ? writer.WriteVector("parent", std::vector<Index>(parent.begin(), parent.end()))
              : writer.WriteVector("parent", parent)) ||
        !writer.WriteScalar("base_error", base_error)) {
        if (error) {
            *error = "SaveBaseResultsSTLQ: failed to write datasets.";
        }
        return false;
    }
    if (virt && virt->n_virtual() > 0) {
        const int nlist = virt->nlist;
        std::vector<int> nvirt_by_cluster(static_cast<std::size_t>(nlist), 0);
        for (int cid = 0; cid < nlist; ++cid) {
            nvirt_by_cluster[static_cast<std::size_t>(cid)] = virt->n_virtual(cid);
        }
        if (!writer.WriteVector("virtual/nvirt_by_cluster", nvirt_by_cluster)) {
            if (error) {
                *error = "SaveBaseResultsSTLQ: failed to write virtual datasets.";
            }
            return false;
        }
        for (int cid = 0; cid < nlist; ++cid) {
            const int nvc = nvirt_by_cluster[static_cast<std::size_t>(cid)];
            if (nvc <= 0) {
                continue;
            }
            const auto& Bc = virt->B_by_cluster[static_cast<std::size_t>(cid)];
            const auto& ac = virt->a_by_cluster[static_cast<std::size_t>(cid)];
            if (!writer.WriteMatrix("virtual/B_" + std::to_string(cid), Bc) ||
                !writer.WriteMatrix("virtual/a_" + std::to_string(cid), ac)) {
                if (error) {
                    *error = "SaveBaseResultsSTLQ: failed to write per-cluster virtual datasets.";
                }
                return false;
            }
        }
        writer.WriteScalar("meta/n_base_virtual", virt->n_virtual());
        writer.WriteScalar("meta/virtual/nlist", virt->nlist);
    }
    writer.WriteString("meta/config_snapshot", DumpConfigMini(config));
    if (base_error >= 0.0f) {
        writer.WriteScalar("meta/base/final_mse", base_error);
        writer.WriteScalar("meta/linkage/mse_mean", base_error);
    }
    if (linkage_ratio >= 0.0f) {
        writer.WriteScalar("meta/linkage/ratio", linkage_ratio);
    }
    if (linkage_max_depth >= 0) {
        writer.WriteScalar("meta/linkage/max_depth", linkage_max_depth);
    }
    if (linkage_mean_depth >= 0.0f) {
        writer.WriteScalar("meta/linkage/mean_depth", linkage_mean_depth);
    }
    if (cluster_id && !cluster_id->empty()) {
        if (config.io.index_dtype == "uint") {
            writer.WriteVector("cluster_id",
                               std::vector<Index>(cluster_id->begin(), cluster_id->end()));
        } else {
            writer.WriteVector("cluster_id", *cluster_id);
        }
    }
    if (n_base_real >= 0) {
        writer.WriteScalar("meta/n_base_real", n_base_real);
    }
    if (is_bad_cluster && !is_bad_cluster->empty()) {
        const std::vector<std::uint8_t> is_bad_u8 = BoolToU8(*is_bad_cluster);
        writer.WriteVector("meta/is_bad_cluster", is_bad_u8);
    }
    return true;
#endif
}

bool LoadBaseResultsHq(const std::string& base_path,
                       const std::string& load_date,
                       const std::string& load_seq,
                       BaseEncoding* base,
                       int* n_base_real,
                       VirtualEncoding* virt,
                       std::string* error) {
    if (!base) {
        if (error) {
            *error = "LoadBaseResultsHq: BaseEncoding is null.";
        }
        return false;
    }
    const std::string path = MakeDatedPathForLoad(base_path, load_date, load_seq, error);
    if (path.empty()) {
        return false;
    }
    LogInfo("Loading base results from: " + path);
    if (ChooseResultLoadFormat(path) == ResultStorageFormat::kBin) {
        return LoadBaseResultsHqBinImpl(path, base, n_base_real, virt, error);
    }
#if !(defined(STLQ_ENABLE_HDF5) && STLQ_ENABLE_HDF5)
    if (error) {
        *error = "LoadBaseResultsHq: HDF5 support is disabled and file is not a native STLQ bin result.";
    }
    return false;
#else
    Hdf5Reader reader(path);
    if (!reader.IsOpen()) {
        if (error) {
            *error = "LoadBaseResultsHq: failed to open " + path;
        }
        return false;
    }
    std::string prefix;
    if (reader.Has("B_base")) {
        prefix.clear();
    } else if (reader.Has("1/B_base")) {
        prefix = "1/";
    }
    if (!reader.ReadMatrix(prefix + "B_base", &base->B) ||
        !reader.ReadMatrix(prefix + "a_base", &base->a)) {
        if (error) {
            *error = "LoadBaseResultsHq: missing datasets under prefix '" + prefix + "'";
        }
        return false;
    }
    if (n_base_real) {
        *n_base_real = -1;
        int tmp = 0;
        if (reader.Has("meta/n_base_real") && reader.ReadScalar("meta/n_base_real", &tmp)) {
            *n_base_real = tmp;
        }
    }
    if (virt) {
        *virt = {};
        if (reader.Has(prefix + "virtual/nvirt_by_cluster")) {
            std::vector<int> nvirt_by_cluster;
            if (!reader.ReadVector(prefix + "virtual/nvirt_by_cluster", &nvirt_by_cluster)) {
                if (error) {
                    *error = "LoadBaseResultsHq: failed to read nvirt_by_cluster.";
                }
                return false;
            }
            const int nlist = static_cast<int>(nvirt_by_cluster.size());
            virt->m = base->B.rows;
            virt->nlist = nlist;
            virt->B_by_cluster.assign(static_cast<std::size_t>(nlist), {});
            virt->a_by_cluster.assign(static_cast<std::size_t>(nlist), {});
            for (int cid = 0; cid < nlist; ++cid) {
                const int nvc = nvirt_by_cluster[static_cast<std::size_t>(cid)];
                if (nvc <= 0) {
                    continue;
                }
                if (!reader.ReadMatrix(prefix + "virtual/B_" + std::to_string(cid),
                                       &virt->B_by_cluster[static_cast<std::size_t>(cid)]) ||
                    !reader.ReadMatrix(prefix + "virtual/a_" + std::to_string(cid),
                                       &virt->a_by_cluster[static_cast<std::size_t>(cid)])) {
                    if (error) {
                        *error = "LoadBaseResultsHq: failed to read per-cluster virtual datasets.";
                    }
                    return false;
                }
            }
        } else if (reader.Has(prefix + "B_virtual")) {
            // Backward-compatible: old format stored a single concatenated matrix + offsets.
            ColMajorMatrix<FullCode> B_flat;
            ColMajorMatrix<float> a_flat;
            std::vector<int> offsets;
            if (!reader.ReadMatrix(prefix + "B_virtual", &B_flat) ||
                !reader.ReadMatrix(prefix + "a_virtual", &a_flat) ||
                !reader.ReadVector(prefix + "virtual_offsets_by_cluster", &offsets)) {
                if (error) {
                    *error = "LoadBaseResultsHq: failed to read legacy virtual datasets.";
                }
                return false;
            }
            const int nlist = static_cast<int>(offsets.size()) - 1;
            virt->m = B_flat.rows;
            virt->nlist = nlist;
            virt->B_by_cluster.assign(static_cast<std::size_t>(nlist), {});
            virt->a_by_cluster.assign(static_cast<std::size_t>(nlist), {});
            for (int cid = 0; cid < nlist; ++cid) {
                const int off = offsets[static_cast<std::size_t>(cid)];
                const int nvc = offsets[static_cast<std::size_t>(cid + 1)] - off;
                if (nvc <= 0) {
                    continue;
                }
                virt->B_by_cluster[static_cast<std::size_t>(cid)] = ColMajorMatrix<FullCode>(virt->m, nvc);
                virt->a_by_cluster[static_cast<std::size_t>(cid)] = ColMajorMatrix<float>(virt->m, nvc);
                std::memcpy(virt->B_by_cluster[static_cast<std::size_t>(cid)].data.data(),
                            B_flat.data.data() + static_cast<std::size_t>(off) * virt->m,
                            sizeof(FullCode) * static_cast<std::size_t>(virt->m) * static_cast<std::size_t>(nvc));
                std::memcpy(virt->a_by_cluster[static_cast<std::size_t>(cid)].data.data(),
                            a_flat.data.data() + static_cast<std::size_t>(off) * virt->m,
                            sizeof(float) * static_cast<std::size_t>(virt->m) * static_cast<std::size_t>(nvc));
            }
        }
    }
    return true;
#endif
}

bool LoadBaseResultsSTLQ(const std::string& base_path,
                           const std::string& load_date,
                           const std::string& load_seq,
                           const IOConfig& io_cfg,
                           BaseEncoding* base,
                           std::vector<int>* parent,
                           std::vector<int>* cluster_id,
                           int* n_base_real,
                           VirtualEncoding* virt,
                           std::string* error) {
    if (!base || !parent) {
        if (error) {
            *error = "LoadBaseResultsSTLQ: output pointers are null.";
        }
        return false;
    }
    (void)io_cfg;
    const std::string path = MakeDatedPathForLoad(base_path, load_date, load_seq, error);
    if (path.empty()) {
        return false;
    }
    LogInfo("Loading linkage results from: " + path);
    if (ChooseResultLoadFormat(path) == ResultStorageFormat::kBin) {
        return LoadBaseResultsSTLQBinImpl(path, base, parent, cluster_id, n_base_real, virt, error);
    }
#if !(defined(STLQ_ENABLE_HDF5) && STLQ_ENABLE_HDF5)
    if (error) {
        *error = "LoadBaseResultsSTLQ: HDF5 support is disabled and file is not a native STLQ bin result.";
    }
    return false;
#else
    Hdf5Reader reader(path);
    if (!reader.IsOpen()) {
        if (error) {
            *error = "LoadBaseResultsSTLQ: failed to open " + path;
        }
        return false;
    }
    std::string prefix;
    if (reader.Has("B_base")) {
        prefix.clear();
    } else if (reader.Has("1/B_base")) {
        prefix = "1/";
    }
    if (!reader.ReadMatrix(prefix + "B_base", &base->B) ||
        !reader.ReadMatrix(prefix + "a_base", &base->a)) {
        if (error) {
            *error = "LoadBaseResultsSTLQ: missing datasets under prefix '" + prefix + "'";
        }
        return false;
    }
    parent->clear();
    if (io_cfg.index_dtype == "uint") {
        std::vector<Index> tmp;
        if (!reader.ReadVector(prefix + "parent", &tmp)) {
            if (error) {
                *error = "LoadBaseResultsSTLQ: missing parent under prefix '" + prefix + "'";
            }
            return false;
        }
        parent->assign(tmp.begin(), tmp.end());
    } else {
        if (!reader.ReadVector(prefix + "parent", parent)) {
            if (error) {
                *error = "LoadBaseResultsSTLQ: missing parent under prefix '" + prefix + "'";
            }
            return false;
        }
    }
    if (cluster_id) {
        cluster_id->clear();
        if (reader.Has(prefix + "cluster_id")) {
            if (io_cfg.index_dtype == "uint") {
                std::vector<Index> tmp;
                if (reader.ReadVector(prefix + "cluster_id", &tmp)) {
                    cluster_id->assign(tmp.begin(), tmp.end());
                }
            } else {
                reader.ReadVector(prefix + "cluster_id", cluster_id);
            }
        }
    }
    if (n_base_real) {
        *n_base_real = -1;
        int tmp = 0;
        if (reader.Has("meta/n_base_real") && reader.ReadScalar("meta/n_base_real", &tmp)) {
            *n_base_real = tmp;
        }
    }
    if (virt) {
        *virt = {};
        if (reader.Has(prefix + "virtual/nvirt_by_cluster")) {
            std::vector<int> nvirt_by_cluster;
            if (!reader.ReadVector(prefix + "virtual/nvirt_by_cluster", &nvirt_by_cluster)) {
                if (error) {
                    *error = "LoadBaseResultsSTLQ: failed to read nvirt_by_cluster.";
                }
                return false;
            }
            const int nlist = static_cast<int>(nvirt_by_cluster.size());
            virt->m = base->B.rows;
            virt->nlist = nlist;
            virt->B_by_cluster.assign(static_cast<std::size_t>(nlist), {});
            virt->a_by_cluster.assign(static_cast<std::size_t>(nlist), {});
            for (int cid = 0; cid < nlist; ++cid) {
                const int nvc = nvirt_by_cluster[static_cast<std::size_t>(cid)];
                if (nvc <= 0) {
                    continue;
                }
                if (!reader.ReadMatrix(prefix + "virtual/B_" + std::to_string(cid),
                                       &virt->B_by_cluster[static_cast<std::size_t>(cid)]) ||
                    !reader.ReadMatrix(prefix + "virtual/a_" + std::to_string(cid),
                                       &virt->a_by_cluster[static_cast<std::size_t>(cid)])) {
                    if (error) {
                        *error = "LoadBaseResultsSTLQ: failed to read per-cluster virtual datasets.";
                    }
                    return false;
                }
            }
        } else if (reader.Has(prefix + "B_virtual")) {
            // Backward-compatible: old format stored a single concatenated matrix + offsets.
            ColMajorMatrix<FullCode> B_flat;
            ColMajorMatrix<float> a_flat;
            std::vector<int> offsets;
            if (!reader.ReadMatrix(prefix + "B_virtual", &B_flat) ||
                !reader.ReadMatrix(prefix + "a_virtual", &a_flat) ||
                !reader.ReadVector(prefix + "virtual_offsets_by_cluster", &offsets)) {
                if (error) {
                    *error = "LoadBaseResultsSTLQ: failed to read legacy virtual datasets.";
                }
                return false;
            }
            const int nlist = static_cast<int>(offsets.size()) - 1;
            virt->m = B_flat.rows;
            virt->nlist = nlist;
            virt->B_by_cluster.assign(static_cast<std::size_t>(nlist), {});
            virt->a_by_cluster.assign(static_cast<std::size_t>(nlist), {});
            for (int cid = 0; cid < nlist; ++cid) {
                const int off = offsets[static_cast<std::size_t>(cid)];
                const int nvc = offsets[static_cast<std::size_t>(cid + 1)] - off;
                if (nvc <= 0) {
                    continue;
                }
                virt->B_by_cluster[static_cast<std::size_t>(cid)] = ColMajorMatrix<FullCode>(virt->m, nvc);
                virt->a_by_cluster[static_cast<std::size_t>(cid)] = ColMajorMatrix<float>(virt->m, nvc);
                std::memcpy(virt->B_by_cluster[static_cast<std::size_t>(cid)].data.data(),
                            B_flat.data.data() + static_cast<std::size_t>(off) * virt->m,
                            sizeof(FullCode) * static_cast<std::size_t>(virt->m) * static_cast<std::size_t>(nvc));
                std::memcpy(virt->a_by_cluster[static_cast<std::size_t>(cid)].data.data(),
                            a_flat.data.data() + static_cast<std::size_t>(off) * virt->m,
                            sizeof(float) * static_cast<std::size_t>(virt->m) * static_cast<std::size_t>(nvc));
            }
        }
    }
    return true;
#endif
}

}  // namespace stlq::io
