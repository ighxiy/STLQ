#include "stlq/io/dataset_io.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "stlq/io/fbin_reader.h"
#include "stlq/io/ibin_reader.h"
#include "stlq/common/logger.h"

namespace stlq::io {

namespace {

struct DatasetPaths {
    std::string train;
    std::string base;
    std::string query;
    std::string groundtruth;
};

std::string JoinPath(const std::string& root, const std::string& rel) {
    if (root.empty()) {
        return rel;
    }
    if (root.back() == '/') {
        return root + rel;
    }
    return root + "/" + rel;
}

std::string JoinPathMaybeAbsolute(const std::string& root, const std::string& p) {
    if (p.empty()) return {};
    const std::filesystem::path path(p);
    if (path.is_absolute()) {
        return p;
    }
    // Treat "C:\..." as absolute on Windows even if `is_absolute()` is false in some configurations.
    if (p.size() >= 2u && std::isalpha(static_cast<unsigned char>(p[0])) && p[1] == ':') {
        return p;
    }
    return JoinPath(root, p);
}

std::string FirstExistingPath(const std::vector<std::string>& candidates) {
    std::error_code ec;
    for (const std::string& candidate : candidates) {
        if (candidate.empty()) {
            continue;
        }
        if (std::filesystem::exists(candidate, ec) && !ec) {
            return candidate;
        }
        ec.clear();
    }
    return {};
}

bool HasSuffix(const std::string& s, const std::string& suffix) {
    if (s.size() < suffix.size()) {
        return false;
    }
    return s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

int PickBigannGroundtruthSubsetM(std::int64_t nbase) {
    // BigANN provides gt for subsets (first N vectors) where N in:
    // {1M,2M,5M,10M,20M,50M,100M,200M,500M,1000M}.
    // We pick the largest available subset <= nbase to keep ids valid.
    // If nbase is not an exact match, recall is not strictly comparable; caller may warn.
    static constexpr int kChoices[] = {1, 2, 5, 10, 20, 50, 100, 200, 500, 1000};
    const std::int64_t nbase_m = nbase / 1'000'000;
    int best = kChoices[0];
    for (int m : kChoices) {
        if (nbase_m >= m) {
            best = m;
        } else {
            break;
        }
    }
    return best;
}

std::string FormatDouble(double value, int precision) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(precision);
    oss << value;
    return oss.str();
}

bool ResolvePaths(const DatasetConfig& config, DatasetPaths* paths, std::string* error) {
    if (!paths) {
        return false;
    }
    const std::string& root = config.data_root;
    const std::string& name = config.name;
    bool known = true;
    if (name == "SIFT1M") {
        paths->train = JoinPath(root, "sift/sift_learn.fvecs");
        paths->base = JoinPath(root, "sift/sift_base.fvecs");
        paths->query = JoinPath(root, "sift/sift_query.fvecs");
        paths->groundtruth = JoinPath(root, "sift/sift_groundtruth.ivecs");
    } else if (name == "SIFTSMALL") {
        paths->train = JoinPath(root, "siftsmall/siftsmall_learn.fvecs");
        paths->base = JoinPath(root, "siftsmall/siftsmall_base.fvecs");
        paths->query = JoinPath(root, "siftsmall/siftsmall_query.fvecs");
        paths->groundtruth = JoinPath(root, "siftsmall/siftsmall_groundtruth.ivecs");
    } else if (name == "GIST1M") {
        paths->train = JoinPath(root, "gist/gist_learn.fvecs");
        paths->base = JoinPath(root, "gist/gist_base.fvecs");
        paths->query = JoinPath(root, "gist/gist_query.fvecs");
        paths->groundtruth = JoinPath(root, "gist/gist_groundtruth.ivecs");
    } else if (name == "SIFT1B") {
        // BigANN / ANN_SIFT1B layout (as shipped in `dataset/bigann/` in this workspace):
        // - Train:   learn.bvecs (100M)
        // - Base:    1milliard.p1.siftbin (1B, bvecs-like records: int32 d + uint8[d])
        // - Query:   queries.bvecs (10k)
        // - GT:      bigann_gnd/idx_{N}M.ivecs (ids are 0-based)
        //
        // Ground-truth is provided for subsets where the first N vectors are used as the base.
        // We pick the largest available N <= requested nbase.
        paths->train = JoinPath(root, "bigann/learn.bvecs");
        paths->base = JoinPath(root, "bigann/1milliard.p1.siftbin");
        paths->query = JoinPath(root, "bigann/queries.bvecs");
        const int subset_m = PickBigannGroundtruthSubsetM(config.nbase);
        paths->groundtruth = JoinPath(root, "bigann/bigann_gnd/idx_" + std::to_string(subset_m) + "M.ivecs");
    } else if (name == "DEEP10M") {
        // Yandex Deep10M (fbin/ibin format, float32 96-d):
        // - Base:    base.10M.fbin (10M)
        // - Train:   same as base (train is sampled from base; paths.train = paths.base)
        // - Query:   query.public.10K.fbin (10k)
        // - GT:      groundtruth.1M.ibin only for base=first 1M;
        //            groundtruth.10M.ibin for base=10M.
        //            Other base subset sizes require an explicit override.
        paths->base = FirstExistingPath({
            JoinPath(root, "base.10M.fbin"),
            JoinPath(root, "deep10m/base.10M.fbin"),
        });
        paths->train = paths->base;  // no separate learn file; train sampled from base
        paths->query = FirstExistingPath({
            JoinPath(root, "query.public.10K.fbin"),
            JoinPath(root, "deep10m/query.public.10K.fbin"),
        });
        if (config.nbase_set && config.nbase == 1000000) {
            paths->groundtruth = FirstExistingPath({
                JoinPath(root, "groundtruth.1M.ibin"),
                JoinPath(root, "deep10m/groundtruth.1M.ibin"),
            });
        } else if (!config.nbase_set || config.nbase == 10000000) {
            paths->groundtruth = FirstExistingPath({
                JoinPath(root, "groundtruth.10M.ibin"),
                JoinPath(root, "deep10m/groundtruth.10M.ibin"),
            });
        } else {
            if (error) {
                *error = "DEEP10M only has built-in groundtruth for dataset.nbase=1000000 or 10000000; "
                         "set dataset.groundtruth_path explicitly for other subsets.";
            }
            return false;
        }
    } else if (name == "MSONG") {
        // Million Song features (fvecs/ivecs format, float32 420-d):
        // - Base:    msong_base.fvecs (994185)
        // - Train:   same as base by default; use offsets to expose a non-overlapping train/base split.
        // - Query:   msong_query.fvecs (1000)
        // - GT:      msong_groundtruth.ivecs (top-100, 0-based)
        paths->base = FirstExistingPath({
            JoinPath(root, "msong_base.fvecs"),
            JoinPath(root, "msong/msong_base.fvecs"),
        });
        paths->train = paths->base;
        paths->query = FirstExistingPath({
            JoinPath(root, "msong_query.fvecs"),
            JoinPath(root, "msong/msong_query.fvecs"),
        });
        paths->groundtruth = FirstExistingPath({
            JoinPath(root, "msong_groundtruth.ivecs"),
            JoinPath(root, "msong/msong_groundtruth.ivecs"),
        });
    } else if (name == "GLOVE100") {
        // ANN-Benchmarks GloVe100 angular dataset converted to normalized fvecs/ivecs:
        // - Base:    glove100_base.fvecs (1183514, row-L2-normalized)
        // - Train:   same as base by default for simple overlap tests
        // - Query:   glove100_query.fvecs (10000, row-L2-normalized)
        // - GT:      glove100_groundtruth.ivecs (top-100, 0-based)
        paths->base = FirstExistingPath({
            JoinPath(root, "glove100_base.fvecs"),
            JoinPath(root, "glove100/converted/glove100_base.fvecs"),
        });
        paths->train = paths->base;
        paths->query = FirstExistingPath({
            JoinPath(root, "glove100_query.fvecs"),
            JoinPath(root, "glove100/converted/glove100_query.fvecs"),
        });
        paths->groundtruth = FirstExistingPath({
            JoinPath(root, "glove100_groundtruth.ivecs"),
            JoinPath(root, "glove100/converted/glove100_groundtruth.ivecs"),
        });
    } else if (name == "DEEP1M") {
        // FAISS/QINCO Deep1M split (fvecs/ivecs format, float32 96-d):
        // - Train:   learn.fvecs (10M learn vectors; default ntrain still controls how many are used)
        // - Base:    base.fvecs (1M)
        // - Query:   deep1B_queries.fvecs (10k)
        // - GT:      deep1M_groundtruth.ivecs
        paths->train = FirstExistingPath({
            JoinPath(root, "learn.fvecs"),
            JoinPath(root, "deep1m/learn.fvecs"),
        });
        paths->base = FirstExistingPath({
            JoinPath(root, "base.fvecs"),
            JoinPath(root, "deep1m/base.fvecs"),
        });
        paths->query = FirstExistingPath({
            JoinPath(root, "deep1B_queries.fvecs"),
            JoinPath(root, "deep1m/deep1B_queries.fvecs"),
        });
        paths->groundtruth = FirstExistingPath({
            JoinPath(root, "deep1M_groundtruth.ivecs"),
            JoinPath(root, "deep1m/deep1M_groundtruth.ivecs"),
        });
    } else {
        known = false;
    }

    // Apply optional overrides (relative to data_root unless absolute).
    const bool train_over = !config.train_path.empty();
    if (!config.base_path.empty()) paths->base = JoinPathMaybeAbsolute(root, config.base_path);
    if (train_over) paths->train = JoinPathMaybeAbsolute(root, config.train_path);
    if (!config.query_path.empty()) paths->query = JoinPathMaybeAbsolute(root, config.query_path);
    if (!config.groundtruth_path.empty()) paths->groundtruth = JoinPathMaybeAbsolute(root, config.groundtruth_path);

    // Datasets without a separate learn file default to train==base unless explicitly overridden.
    if ((name == "DEEP10M" || name == "MSONG" || name == "GLOVE100") && !train_over) {
        paths->train = paths->base;
    }

    if (!known) {
        const bool ok_custom =
            !paths->train.empty() && !paths->base.empty() && !paths->query.empty() && !paths->groundtruth.empty();
        if (!ok_custom) {
            if (error) {
                *error = "Unsupported dataset name: " + name +
                         " (set dataset.{train_path,base_path,query_path,groundtruth_path} to use a custom dataset).";
            }
            return false;
        }
    }

    return true;
}

bool ReadVecsHeader(std::ifstream& in,
                    const std::string& path,
                    std::int32_t* dim,
                    std::int64_t* count,
                    std::size_t elem_size,
                    std::string* error) {
    std::int32_t d = 0;
    in.read(reinterpret_cast<char*>(&d), sizeof(std::int32_t));
    if (!in) {
        if (error) {
            *error = "Failed to read dimension from " + path;
        }
        return false;
    }
    in.seekg(0, std::ios::end);
    std::streamoff size = in.tellg();
    std::int64_t vec_size = static_cast<std::int64_t>(sizeof(std::int32_t)) +
                            static_cast<std::int64_t>(d) * static_cast<std::int64_t>(elem_size);
    if (vec_size <= 0) {
        if (error) {
            *error = "Invalid vector size in " + path;
        }
        return false;
    }
    *count = static_cast<std::int64_t>(size) / vec_size;
    *dim = d;
    in.seekg(0, std::ios::beg);
    return true;
}

[[maybe_unused]] bool ReadFvecs(const std::string& path,
                                int* n_io,
                                ColMajorMatrix<float>* out,
                                std::string* error) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        if (error) {
            *error = "Failed to open file: " + path;
        }
        return false;
    }

    std::int32_t dim = 0;
    std::int64_t count = 0;
    if (!ReadVecsHeader(in, path, &dim, &count, sizeof(float), error)) {
        return false;
    }
    if (!n_io || *n_io <= 0) {
        if (error) {
            *error = "Requested vector count must be positive for " + path;
        }
        return false;
    }
    if (count <= 0) {
        if (error) {
            *error = "No vectors found in " + path;
        }
        return false;
    }
    const int available = static_cast<int>(
        std::min<std::int64_t>(count, static_cast<std::int64_t>(std::numeric_limits<int>::max())));
    int n = std::min(*n_io, available);
    if (*n_io > available) {
        LogWarn("Requested " + std::to_string(*n_io) + " vectors but file has " +
                std::to_string(count) + " for " + path + "; shrinking to " +
                std::to_string(n) + ".");
    }
    *n_io = n;

    out->rows = dim;
    out->cols = n;
    out->data.resize(static_cast<std::size_t>(dim) * n);
    std::vector<float> buffer(dim);

    for (int i = 0; i < n; ++i) {
        std::int32_t d_read = 0;
        in.read(reinterpret_cast<char*>(&d_read), sizeof(std::int32_t));
        if (!in || d_read != dim) {
            if (error) {
                *error = "Dimension mismatch while reading " + path;
            }
            return false;
        }
        in.read(reinterpret_cast<char*>(buffer.data()), sizeof(float) * dim);
        if (!in) {
            if (error) {
                *error = "Failed to read vector data from " + path;
            }
            return false;
        }
        float* dst = out->Col(i);
        std::memcpy(dst, buffer.data(), static_cast<std::size_t>(dim) * sizeof(float));
    }
    return true;
}

[[maybe_unused]] bool ReadBvecsAsF32(const std::string& path,
                                     int* n_io,
                                     ColMajorMatrix<float>* out,
                                     std::string* error) {
    if (!n_io || *n_io <= 0) {
        if (error) {
            *error = "Requested vector count must be positive for " + path;
        }
        return false;
    }
    if (!out) {
        if (error) {
            *error = "ReadBvecsAsF32: output is null for " + path;
        }
        return false;
    }

    BvecsReader reader;
    if (!reader.Open(path, error)) {
        return false;
    }
    const int available = static_cast<int>(
        std::min<std::uint64_t>(reader.n(), static_cast<std::uint64_t>(std::numeric_limits<int>::max())));
    int n = std::min(*n_io, available);
    if (*n_io > available) {
        LogWarn("Requested " + std::to_string(*n_io) + " vectors but file has " +
                std::to_string(reader.n()) + " for " + path + "; shrinking to " +
                std::to_string(n) + ".");
    }
    *n_io = n;

    ColMajorMatrix<std::uint8_t> tmp;
    if (!reader.ReadBlock(0, static_cast<std::uint32_t>(n), &tmp, error)) {
        return false;
    }
    ConvertBvecsToF32(tmp, out);
    return true;
}

[[maybe_unused]] bool ReadFbin(const std::string& path,
                               int* n_io,
                               ColMajorMatrix<float>* out,
                               std::string* error) {
    if (!n_io || *n_io <= 0) {
        if (error) *error = "Requested vector count must be positive for " + path;
        return false;
    }
    if (!out) {
        if (error) *error = "ReadFbin: output is null for " + path;
        return false;
    }

    FbinReader reader;
    if (!reader.Open(path, error)) {
        return false;
    }
    const int available = static_cast<int>(
        std::min<std::uint64_t>(reader.n(), static_cast<std::uint64_t>(std::numeric_limits<int>::max())));
    int n = std::min(*n_io, available);
    if (*n_io > available) {
        LogWarn("Requested " + std::to_string(*n_io) + " vectors but file has " +
                std::to_string(reader.n()) + " for " + path + "; shrinking to " +
                std::to_string(n) + ".");
    }
    *n_io = n;

    if (!reader.ReadBlock(0, static_cast<std::uint32_t>(n), out, error)) {
        return false;
    }
    return true;
}

bool ReadIvecsFirst(const std::string& path,
                    int* n_io,
                    bool add_one,
                    std::vector<int>* out,
                    std::string* error) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        if (error) {
            *error = "Failed to open file: " + path;
        }
        return false;
    }

    std::int32_t dim = 0;
    std::int64_t count = 0;
    if (!ReadVecsHeader(in, path, &dim, &count, sizeof(std::int32_t), error)) {
        return false;
    }
    if (dim <= 0) {
        if (error) {
            *error = "Invalid ivecs dimension in " + path;
        }
        return false;
    }
    if (!n_io || *n_io <= 0) {
        if (error) {
            *error = "Requested vector count must be positive for " + path;
        }
        return false;
    }
    if (count <= 0) {
        if (error) {
            *error = "No vectors found in " + path;
        }
        return false;
    }
    const int available = static_cast<int>(
        std::min<std::int64_t>(count, static_cast<std::int64_t>(std::numeric_limits<int>::max())));
    int n = std::min(*n_io, available);
    if (*n_io > available) {
        LogWarn("Requested " + std::to_string(*n_io) + " vectors but file has " +
                std::to_string(count) + " for " + path + "; shrinking to " +
                std::to_string(n) + ".");
    }
    *n_io = n;

    out->resize(n);
    std::vector<std::int32_t> buffer(dim);

    for (int i = 0; i < n; ++i) {
        std::int32_t d_read = 0;
        in.read(reinterpret_cast<char*>(&d_read), sizeof(std::int32_t));
        if (!in || d_read != dim) {
            if (error) {
                *error = "Dimension mismatch while reading " + path;
            }
            return false;
        }
        in.read(reinterpret_cast<char*>(buffer.data()), sizeof(std::int32_t) * dim);
        if (!in) {
            if (error) {
                *error = "Failed to read ivecs data from " + path;
            }
            return false;
        }
        std::int32_t val = buffer[0];
        if (add_one) {
            val += 1;
        }
        (*out)[i] = static_cast<int>(val);
    }
    return true;
}

// Read first neighbor (column 0) from a `.ibin` groundtruth file (flat binary: [nrows, ncols, data...]).
bool ReadIbinFirst(const std::string& path,
                   int* n_io,
                   bool add_one,
                   std::vector<int>* out,
                   std::string* error) {
    if (!n_io || *n_io <= 0 || !out) {
        if (error) *error = "ReadIbinFirst: invalid args for " + path;
        return false;
    }

    IbinReader reader;
    if (!reader.Open(path, error)) {
        return false;
    }
    const int k = reader.k();
    if (k <= 0) {
        if (error) *error = "ReadIbinFirst: zero k in " + path;
        return false;
    }
    const int available = static_cast<int>(
        std::min<std::uint64_t>(reader.n(), static_cast<std::uint64_t>(std::numeric_limits<int>::max())));
    int n = std::min(*n_io, available);
    if (*n_io > available) {
        LogWarn("ReadIbinFirst: requested " + std::to_string(*n_io) + " but file has " +
                std::to_string(reader.n()) + "; shrinking to " + std::to_string(n) + ".");
    }
    *n_io = n;

    ColMajorMatrix<std::int32_t> block;
    if (!reader.ReadBlock(0, static_cast<std::uint32_t>(n), &block, error)) {
        return false;
    }

    out->resize(n);
    for (int i = 0; i < n; ++i) {
        std::int32_t val = block.Col(i)[0];  // first neighbor
        if (add_one) val += 1;
        (*out)[i] = static_cast<int>(val);
    }
    return true;
}

// Auto-dispatch for groundtruth: .ivecs or .ibin.
bool ReadGroundtruthFirst(const std::string& path,
                          int* n_io,
                          bool add_one,
                          std::vector<int>* out,
                          std::string* error) {
    if (HasSuffix(path, ".ivecs")) {
        return ReadIvecsFirst(path, n_io, add_one, out, error);
    }
    if (HasSuffix(path, ".ibin")) {
        return ReadIbinFirst(path, n_io, add_one, out, error);
    }
    if (error) *error = "Unsupported groundtruth file extension for " + path;
    return false;
}

bool ReadGroundtruthTopKIvecs(const std::string& path,
                              int nq,
                              int topk,
                              bool add_one,
                              ColMajorMatrix<int>* gt_topk,
                              std::string* error) {
    if (!gt_topk || nq <= 0 || topk <= 0) {
        if (error) *error = "ReadGroundtruthTopKIvecs: invalid args.";
        return false;
    }
    IvecsReader reader;
    if (!reader.Open(path, error)) {
        return false;
    }
    const int k_file = reader.k();
    if (k_file <= 0) {
        if (error) *error = "ReadGroundtruthTopKIvecs: zero k in " + path;
        return false;
    }
    const int nq_file = static_cast<int>(
        std::min<std::uint64_t>(reader.n(), static_cast<std::uint64_t>(std::numeric_limits<int>::max())));
    if (nq > nq_file) {
        if (error) *error = "ReadGroundtruthTopKIvecs: requested nq=" + std::to_string(nq) +
                            " exceeds file rows=" + std::to_string(nq_file);
        return false;
    }
    const int out_k = std::min(topk, k_file);
    ColMajorMatrix<std::int32_t> raw;
    if (!reader.ReadBlock(0, static_cast<std::uint32_t>(nq), &raw, error)) {
        return false;
    }
    gt_topk->rows = out_k;
    gt_topk->cols = nq;
    gt_topk->data.resize(static_cast<std::size_t>(out_k) * static_cast<std::size_t>(nq));
    for (int q = 0; q < nq; ++q) {
        const std::int32_t* src = raw.Col(q);
        int* dst = gt_topk->Col(q);
        for (int i = 0; i < out_k; ++i) {
            std::int32_t v = src[i];
            if (add_one) v += 1;
            dst[i] = static_cast<int>(v);
        }
    }
    return true;
}

bool ReadGroundtruthTopKIbin(const std::string& path,
                             int nq,
                             int topk,
                             bool add_one,
                             ColMajorMatrix<int>* gt_topk,
                             std::string* error) {
    if (!gt_topk || nq <= 0 || topk <= 0) {
        if (error) *error = "ReadGroundtruthTopKIbin: invalid args.";
        return false;
    }
    IbinReader reader;
    if (!reader.Open(path, error)) {
        return false;
    }
    const int k_file = reader.k();
    if (k_file <= 0) {
        if (error) *error = "ReadGroundtruthTopKIbin: zero k in " + path;
        return false;
    }
    const int nq_file = static_cast<int>(
        std::min<std::uint64_t>(reader.n(), static_cast<std::uint64_t>(std::numeric_limits<int>::max())));
    if (nq > nq_file) {
        if (error) *error = "ReadGroundtruthTopKIbin: requested nq=" + std::to_string(nq) +
                            " exceeds file rows=" + std::to_string(nq_file);
        return false;
    }
    const int out_k = std::min(topk, k_file);
    ColMajorMatrix<std::int32_t> raw;
    if (!reader.ReadBlock(0, static_cast<std::uint32_t>(nq), &raw, error)) {
        return false;
    }
    gt_topk->rows = out_k;
    gt_topk->cols = nq;
    gt_topk->data.resize(static_cast<std::size_t>(out_k) * static_cast<std::size_t>(nq));
    for (int q = 0; q < nq; ++q) {
        const std::int32_t* src = raw.Col(q);
        int* dst = gt_topk->Col(q);
        for (int i = 0; i < out_k; ++i) {
            std::int32_t v = src[i];
            if (add_one) v += 1;
            dst[i] = static_cast<int>(v);
        }
    }
    return true;
}

bool ShouldAddOne(const DatasetConfig& config) {
    return config.groundtruth_add1 != 0;
}

bool ReadVectorsAutoF32(const std::string& path,
                        int* n_io,
                        ColMajorMatrix<float>* out,
                        std::string* error) {
    if (!n_io || *n_io <= 0) {
        if (error) *error = "Requested vector count must be positive for " + path;
        return false;
    }
    if (!out) {
        if (error) *error = "ReadVectorsAutoF32: output is null for " + path;
        return false;
    }
    if (HasSuffix(path, ".fvecs")) {
        FvecsReader reader;
        if (!reader.Open(path, error)) return false;
        int n = std::min<std::uint64_t>(static_cast<std::uint64_t>(*n_io), reader.n());
        if (static_cast<std::uint64_t>(*n_io) > reader.n()) {
            LogWarn("Requested " + std::to_string(*n_io) + " vectors but file has n=" +
                    std::to_string(reader.n()) + " for " + path + "; shrinking to " + std::to_string(n) + ".");
        }
        *n_io = n;
        return reader.ReadBlock(0, static_cast<std::uint32_t>(n), out, error);
    }
    if (HasSuffix(path, ".fbin")) {
        FbinReader reader;
        if (!reader.Open(path, error)) return false;
        int n = std::min<std::uint64_t>(static_cast<std::uint64_t>(*n_io), reader.n());
        if (static_cast<std::uint64_t>(*n_io) > reader.n()) {
            LogWarn("Requested " + std::to_string(*n_io) + " vectors but file has n=" +
                    std::to_string(reader.n()) + " for " + path + "; shrinking to " + std::to_string(n) + ".");
        }
        *n_io = n;
        return reader.ReadBlock(0, static_cast<std::uint32_t>(n), out, error);
    }
    if (HasSuffix(path, ".bvecs") || HasSuffix(path, ".siftbin")) {
        BvecsReader reader;
        if (!reader.Open(path, error)) return false;
        int n = std::min<std::uint64_t>(static_cast<std::uint64_t>(*n_io), reader.n());
        if (static_cast<std::uint64_t>(*n_io) > reader.n()) {
            LogWarn("Requested " + std::to_string(*n_io) + " vectors but file has n=" +
                    std::to_string(reader.n()) + " for " + path + "; shrinking to " + std::to_string(n) + ".");
        }
        *n_io = n;
        ColMajorMatrix<std::uint8_t> tmp;
        if (!reader.ReadBlock(0, static_cast<std::uint32_t>(n), &tmp, error)) return false;
        ConvertBvecsToF32(tmp, out);
        return true;
    }
    if (error) *error = "Unsupported vector file extension for " + path;
    return false;
}

}  // namespace

bool OpenTrainBvecsReader(const Config& config, BvecsReader* out, std::string* error) {
    if (!out) {
        if (error) {
            *error = "OpenTrainBvecsReader: output is null.";
        }
        return false;
    }
    DatasetPaths paths;
    if (!ResolvePaths(config.dataset, &paths, error)) {
        return false;
    }
    return out->Open(paths.train, error);
}

bool OpenBaseBvecsReader(const Config& config, BvecsReader* out, std::string* error) {
    if (!out) {
        if (error) {
            *error = "OpenBaseBvecsReader: output is null.";
        }
        return false;
    }
    DatasetPaths paths;
    if (!ResolvePaths(config.dataset, &paths, error)) {
        return false;
    }
    return out->Open(paths.base, error);
}

bool OpenTrainFvecsReader(const Config& config, FvecsReader* out, std::string* error) {
    if (!out) {
        if (error) {
            *error = "OpenTrainFvecsReader: output is null.";
        }
        return false;
    }
    DatasetPaths paths;
    if (!ResolvePaths(config.dataset, &paths, error)) {
        return false;
    }
    return out->Open(paths.train, error);
}

bool OpenBaseFvecsReader(const Config& config, FvecsReader* out, std::string* error) {
    if (!out) {
        if (error) {
            *error = "OpenBaseFvecsReader: output is null.";
        }
        return false;
    }
    DatasetPaths paths;
    if (!ResolvePaths(config.dataset, &paths, error)) {
        return false;
    }
    return out->Open(paths.base, error);
}

bool OpenQueryFvecsReader(const Config& config, FvecsReader* out, std::string* error) {
    if (!out) {
        if (error) {
            *error = "OpenQueryFvecsReader: output is null.";
        }
        return false;
    }
    DatasetPaths paths;
    if (!ResolvePaths(config.dataset, &paths, error)) {
        return false;
    }
    return out->Open(paths.query, error);
}

bool OpenTrainFbinReader(const Config& config, FbinReader* out, std::string* error) {
    if (!out) {
        if (error) *error = "OpenTrainFbinReader: output is null.";
        return false;
    }
    DatasetPaths paths;
    if (!ResolvePaths(config.dataset, &paths, error)) {
        return false;
    }
    return out->Open(paths.train, error);
}

bool OpenBaseFbinReader(const Config& config, FbinReader* out, std::string* error) {
    if (!out) {
        if (error) *error = "OpenBaseFbinReader: output is null.";
        return false;
    }
    DatasetPaths paths;
    if (!ResolvePaths(config.dataset, &paths, error)) {
        return false;
    }
    return out->Open(paths.base, error);
}

bool OpenQueryFbinReader(const Config& config, FbinReader* out, std::string* error) {
    if (!out) {
        if (error) *error = "OpenQueryFbinReader: output is null.";
        return false;
    }
    DatasetPaths paths;
    if (!ResolvePaths(config.dataset, &paths, error)) {
        return false;
    }
    return out->Open(paths.query, error);
}

bool OpenQueryBvecsReader(const Config& config, BvecsReader* out, std::string* error) {
    if (!out) {
        if (error) {
            *error = "OpenQueryBvecsReader: output is null.";
        }
        return false;
    }
    DatasetPaths paths;
    if (!ResolvePaths(config.dataset, &paths, error)) {
        return false;
    }
    return out->Open(paths.query, error);
}

bool OpenGroundtruthIvecsReader(const Config& config, IvecsReader* out, std::string* error) {
    if (!out) {
        if (error) {
            *error = "OpenGroundtruthIvecsReader: output is null.";
        }
        return false;
    }
    DatasetPaths paths;
    if (!ResolvePaths(config.dataset, &paths, error)) {
        return false;
    }
    return out->Open(paths.groundtruth, error);
}

bool OpenGroundtruthIbinReader(const Config& config, IbinReader* out, std::string* error) {
    if (!out) {
        if (error) {
            *error = "OpenGroundtruthIbinReader: output is null.";
        }
        return false;
    }
    DatasetPaths paths;
    if (!ResolvePaths(config.dataset, &paths, error)) {
        return false;
    }
    return out->Open(paths.groundtruth, error);
}

void ConvertBvecsToF32(const ColMajorMatrix<std::uint8_t>& in, ColMajorMatrix<float>* out) {
    const int d = in.rows;
    const int n = in.cols;
    out->rows = d;
    out->cols = n;
    out->data.resize(static_cast<std::size_t>(d) * static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const std::uint8_t* src = in.Col(i);
        float* dst = out->Col(i);
        #pragma omp simd
        for (int r = 0; r < d; ++r) {
            dst[r] = static_cast<float>(src[r]);
        }
    }
}

namespace {

std::int64_t ClampTrainCount(const Config& cfg, std::uint64_t file_n, std::int64_t ntrain_req) {
    std::uint64_t n = file_n;
    if (cfg.dataset.ntrain_set && cfg.dataset.ntrain > 0) {
        n = std::min<std::uint64_t>(n, static_cast<std::uint64_t>(cfg.dataset.ntrain));
    }
    if (ntrain_req > 0) {
        n = std::min<std::uint64_t>(n, static_cast<std::uint64_t>(ntrain_req));
    }
    return static_cast<std::int64_t>(n);
}

int ClampSampleCount(std::int64_t ntrain, int K) {
    if (ntrain <= 0 || K <= 0) {
        return 0;
    }
    if (static_cast<std::int64_t>(K) >= ntrain) {
        return static_cast<int>(
            std::min<std::int64_t>(ntrain, static_cast<std::int64_t>(std::numeric_limits<int>::max())));
    }
    return K;
}

}  // namespace

bool ReservoirSampleTrainU8(const Config& cfg,
                            std::int64_t ntrain_req,
                            int K,
                            ColMajorMatrix<std::uint8_t>* Xt_u8_sample,
                            std::string* err) {
    if (!Xt_u8_sample) {
        if (err) *err = "ReservoirSampleTrainU8: output is null.";
        return false;
    }

    BvecsReader reader;
    if (!OpenTrainBvecsReader(cfg, &reader, err)) {
        return false;
    }
    const int d = reader.d();
    // If train==base (no separate learn file), cap train scan by dataset.nbase for subset experiments.
    std::uint64_t train_file_n = reader.n();
    {
        DatasetPaths paths;
        std::string local_err;
        if (ResolvePaths(cfg.dataset, &paths, &local_err) &&
            (paths.train == paths.base) &&
            cfg.dataset.nbase_set && cfg.dataset.nbase > 0) {
            train_file_n = std::min<std::uint64_t>(train_file_n, static_cast<std::uint64_t>(cfg.dataset.nbase));
        }
    }
    const std::int64_t ntrain = ClampTrainCount(cfg, train_file_n, ntrain_req);
    if (d <= 0 || ntrain <= 0) {
        if (err) *err = "ReservoirSampleTrainU8: empty train reader.";
        return false;
    }
    if (ntrain_req > 0 && train_file_n < static_cast<std::uint64_t>(ntrain_req)) {
        LogWarn("ReservoirSampleTrainU8: requested ntrain=" + std::to_string(ntrain_req) +
                " but file has n=" + std::to_string(train_file_n) + "; shrinking to " +
                std::to_string(ntrain) + ".");
    }

    const int K_use = ClampSampleCount(ntrain, K);
    if (K_use <= 0) {
        if (err) *err = "ReservoirSampleTrainU8: invalid sample size.";
        return false;
    }

    Xt_u8_sample->rows = d;
    Xt_u8_sample->cols = K_use;
    Xt_u8_sample->data.resize(static_cast<std::size_t>(d) * static_cast<std::size_t>(K_use));

    std::mt19937_64 rng(static_cast<std::uint64_t>(cfg.train.seed));
    std::uniform_int_distribution<std::uint64_t> dist;

    const std::uint32_t block = static_cast<std::uint32_t>(std::max(1, cfg.large.train_block));
    ColMajorMatrix<std::uint8_t> buf;

    std::uint64_t filled = 0;
    std::uint64_t global = 0;
    while (global < static_cast<std::uint64_t>(ntrain)) {
        const std::uint64_t remaining = static_cast<std::uint64_t>(ntrain) - global;
        const std::uint32_t take = static_cast<std::uint32_t>(std::min<std::uint64_t>(remaining, block));
        if (!reader.ReadBlock(global, take, &buf, err)) {
            return false;
        }
        for (std::uint32_t j = 0; j < take; ++j) {
            const std::uint64_t i = global + j;
            std::uint64_t slot = 0;
            if (filled < static_cast<std::uint64_t>(K_use)) {
                slot = filled++;
            } else {
                slot = dist(rng, std::uniform_int_distribution<std::uint64_t>::param_type(0, i));
                if (slot >= static_cast<std::uint64_t>(K_use)) {
                    continue;
                }
            }
            std::uint8_t* dst = Xt_u8_sample->Col(static_cast<int>(slot));
            const std::uint8_t* src = buf.Col(static_cast<int>(j));
            std::memcpy(dst, src, static_cast<std::size_t>(d) * sizeof(std::uint8_t));
        }
        global += take;
    }

    if (filled < static_cast<std::uint64_t>(K_use)) {
        Xt_u8_sample->cols = static_cast<int>(filled);
        Xt_u8_sample->data.resize(static_cast<std::size_t>(d) * static_cast<std::size_t>(filled));
    }

    double mean = 0.0;
    double m2 = 0.0;
    std::uint64_t cnt = 0;
    for (std::size_t idx = 0; idx < Xt_u8_sample->data.size(); ++idx) {
        const auto x = static_cast<double>(Xt_u8_sample->data[idx]);
        ++cnt;
        const double delta = x - mean;
        mean += delta / static_cast<double>(cnt);
        m2 += delta * (x - mean);
    }
    const double var = (cnt > 1) ? (m2 / static_cast<double>(cnt - 1)) : 0.0;
    LogInfo("Reservoir sample train(u8): ntrain=" + std::to_string(ntrain) +
            " K=" + std::to_string(K_use) + " d=" + std::to_string(d) +
            " mean=" + FormatDouble(mean, 6) +
            " var=" + FormatDouble(var, 6));
    return true;
}

bool ReservoirSampleTrainF32(const Config& cfg,
                             std::int64_t ntrain_req,
                             int K,
                             ColMajorMatrix<float>* Xt_sample,
                             std::string* err) {
    if (!Xt_sample) {
        if (err) *err = "ReservoirSampleTrainF32: output is null.";
        return false;
    }

    FvecsReader reader;
    if (!OpenTrainFvecsReader(cfg, &reader, err)) {
        return false;
    }
    const int d = reader.d();
    // If train==base (no separate learn file), cap train scan by dataset.nbase for subset experiments.
    std::uint64_t train_file_n = reader.n();
    {
        DatasetPaths paths;
        std::string local_err;
        if (ResolvePaths(cfg.dataset, &paths, &local_err) &&
            (paths.train == paths.base) &&
            cfg.dataset.nbase_set && cfg.dataset.nbase > 0) {
            train_file_n = std::min<std::uint64_t>(train_file_n, static_cast<std::uint64_t>(cfg.dataset.nbase));
        }
    }
    const std::int64_t ntrain = ClampTrainCount(cfg, train_file_n, ntrain_req);
    if (d <= 0 || ntrain <= 0) {
        if (err) *err = "ReservoirSampleTrainF32: empty train reader.";
        return false;
    }
    if (ntrain_req > 0 && train_file_n < static_cast<std::uint64_t>(ntrain_req)) {
        LogWarn("ReservoirSampleTrainF32: requested ntrain=" + std::to_string(ntrain_req) +
                " but file has n=" + std::to_string(train_file_n) + "; shrinking to " +
                std::to_string(ntrain) + ".");
    }

    const int K_use = ClampSampleCount(ntrain, K);
    if (K_use <= 0) {
        if (err) *err = "ReservoirSampleTrainF32: invalid sample size.";
        return false;
    }

    Xt_sample->rows = d;
    Xt_sample->cols = K_use;
    Xt_sample->data.resize(static_cast<std::size_t>(d) * static_cast<std::size_t>(K_use));

    std::mt19937_64 rng(static_cast<std::uint64_t>(cfg.train.seed));
    std::uniform_int_distribution<std::uint64_t> dist;

    const std::uint32_t block = static_cast<std::uint32_t>(std::max(1, cfg.large.train_block));
    ColMajorMatrix<float> buf;

    std::uint64_t filled = 0;
    std::uint64_t global = 0;
    while (global < static_cast<std::uint64_t>(ntrain)) {
        const std::uint64_t remaining = static_cast<std::uint64_t>(ntrain) - global;
        const std::uint32_t take = static_cast<std::uint32_t>(std::min<std::uint64_t>(remaining, block));
        if (!reader.ReadBlock(global, take, &buf, err)) {
            return false;
        }
        for (std::uint32_t j = 0; j < take; ++j) {
            const std::uint64_t i = global + j;
            std::uint64_t slot = 0;
            if (filled < static_cast<std::uint64_t>(K_use)) {
                slot = filled++;
            } else {
                slot = dist(rng, std::uniform_int_distribution<std::uint64_t>::param_type(0, i));
                if (slot >= static_cast<std::uint64_t>(K_use)) {
                    continue;
                }
            }
            float* dst = Xt_sample->Col(static_cast<int>(slot));
            const float* src = buf.Col(static_cast<int>(j));
            std::memcpy(dst, src, static_cast<std::size_t>(d) * sizeof(float));
        }
        global += take;
    }

    if (filled < static_cast<std::uint64_t>(K_use)) {
        Xt_sample->cols = static_cast<int>(filled);
        Xt_sample->data.resize(static_cast<std::size_t>(d) * static_cast<std::size_t>(filled));
    }

    LogInfo("Reservoir sample train(f32): ntrain=" + std::to_string(ntrain) +
            " K=" + std::to_string(K_use) + " d=" + std::to_string(d));
    return true;
}

bool ReservoirSampleTrainFbin(const Config& cfg,
                              std::int64_t ntrain_req,
                              int K,
                              ColMajorMatrix<float>* Xt_sample,
                              std::string* err) {
    if (!Xt_sample) {
        if (err) *err = "ReservoirSampleTrainFbin: output is null.";
        return false;
    }

    FbinReader reader;
    if (!OpenTrainFbinReader(cfg, &reader, err)) {
        return false;
    }
    const int d = reader.d();
    // If train==base (no separate learn file), cap train scan by dataset.nbase for subset experiments.
    std::uint64_t train_file_n = reader.n();
    {
        DatasetPaths paths;
        std::string local_err;
        if (ResolvePaths(cfg.dataset, &paths, &local_err) &&
            (paths.train == paths.base) &&
            cfg.dataset.nbase_set && cfg.dataset.nbase > 0) {
            train_file_n = std::min<std::uint64_t>(train_file_n, static_cast<std::uint64_t>(cfg.dataset.nbase));
        }
    }
    const std::int64_t ntrain = ClampTrainCount(cfg, train_file_n, ntrain_req);
    if (d <= 0 || ntrain <= 0) {
        if (err) *err = "ReservoirSampleTrainFbin: empty train reader.";
        return false;
    }
    if (ntrain_req > 0 && train_file_n < static_cast<std::uint64_t>(ntrain_req)) {
        LogWarn("ReservoirSampleTrainFbin: requested ntrain=" + std::to_string(ntrain_req) +
                " but file has n=" + std::to_string(train_file_n) + "; shrinking to " +
                std::to_string(ntrain) + ".");
    }

    const int K_use = ClampSampleCount(ntrain, K);
    if (K_use <= 0) {
        if (err) *err = "ReservoirSampleTrainFbin: invalid sample size.";
        return false;
    }

    Xt_sample->rows = d;
    Xt_sample->cols = K_use;
    Xt_sample->data.resize(static_cast<std::size_t>(d) * static_cast<std::size_t>(K_use));

    std::mt19937_64 rng(static_cast<std::uint64_t>(cfg.train.seed));
    std::uniform_int_distribution<std::uint64_t> dist;

    const std::uint32_t block = static_cast<std::uint32_t>(std::max(1, cfg.large.train_block));
    ColMajorMatrix<float> buf;

    std::uint64_t filled = 0;
    std::uint64_t global = 0;
    while (global < static_cast<std::uint64_t>(ntrain)) {
        const std::uint64_t remaining = static_cast<std::uint64_t>(ntrain) - global;
        const std::uint32_t take = static_cast<std::uint32_t>(std::min<std::uint64_t>(remaining, block));
        if (!reader.ReadBlock(global, take, &buf, err)) {
            return false;
        }
        for (std::uint32_t j = 0; j < take; ++j) {
            const std::uint64_t i = global + j;
            std::uint64_t slot = 0;
            if (filled < static_cast<std::uint64_t>(K_use)) {
                slot = filled++;
            } else {
                slot = dist(rng, std::uniform_int_distribution<std::uint64_t>::param_type(0, i));
                if (slot >= static_cast<std::uint64_t>(K_use)) {
                    continue;
                }
            }
            float* dst = Xt_sample->Col(static_cast<int>(slot));
            const float* src = buf.Col(static_cast<int>(j));
            std::memcpy(dst, src, static_cast<std::size_t>(d) * sizeof(float));
        }
        global += take;
    }

    if (filled < static_cast<std::uint64_t>(K_use)) {
        Xt_sample->cols = static_cast<int>(filled);
        Xt_sample->data.resize(static_cast<std::size_t>(d) * static_cast<std::size_t>(filled));
    }

    LogInfo("Reservoir sample train(fbin): ntrain=" + std::to_string(ntrain) +
            " K=" + std::to_string(K_use) + " d=" + std::to_string(d));
    return true;
}

bool ReservoirSampleTrainToF32(const Config& cfg,
                               std::int64_t ntrain_req,
                               int K,
                               ColMajorMatrix<float>* Xt_sample,
                               std::string* err) {
    if (!Xt_sample) {
        if (err) *err = "ReservoirSampleTrainToF32: output is null.";
        return false;
    }

    const std::string fmt = cfg.large.train_format;
    if (fmt == "fvecs") {
        return ReservoirSampleTrainF32(cfg, ntrain_req, K, Xt_sample, err);
    }
    if (fmt == "fbin") {
        return ReservoirSampleTrainFbin(cfg, ntrain_req, K, Xt_sample, err);
    }
    if (fmt == "bvecs") {
        ColMajorMatrix<std::uint8_t> tmp_u8;
        if (!ReservoirSampleTrainU8(cfg, ntrain_req, K, &tmp_u8, err)) {
            return false;
        }
        ConvertBvecsToF32(tmp_u8, Xt_sample);
        return true;
    }

    // auto: try fvecs, then fbin, then bvecs.
    {
        ColMajorMatrix<float> tmp;
        std::string local_err;
        if (ReservoirSampleTrainF32(cfg, ntrain_req, K, &tmp, &local_err)) {
            *Xt_sample = std::move(tmp);
            return true;
        }
    }
    {
        ColMajorMatrix<float> tmp;
        std::string local_err;
        if (ReservoirSampleTrainFbin(cfg, ntrain_req, K, &tmp, &local_err)) {
            *Xt_sample = std::move(tmp);
            return true;
        }
    }
    ColMajorMatrix<std::uint8_t> tmp_u8;
    if (!ReservoirSampleTrainU8(cfg, ntrain_req, K, &tmp_u8, err)) {
        return false;
    }
    ConvertBvecsToF32(tmp_u8, Xt_sample);
    return true;
}

bool LoadTrainSet(const DatasetConfig& config, ColMajorMatrix<float>* Xt, std::string* error) {
    if (!Xt) {
        if (error) {
            *error = "LoadTrainSet: output is null.";
        }
        return false;
    }
    DatasetPaths paths;
    if (!ResolvePaths(config, &paths, error)) {
        return false;
    }
    int n = config.ntrain;
    return ReadVectorsAutoF32(paths.train, &n, Xt, error);
}

bool LoadBaseSet(const DatasetConfig& config, ColMajorMatrix<float>* Xb, std::string* error) {
    if (!Xb) {
        if (error) {
            *error = "LoadBaseSet: output is null.";
        }
        return false;
    }
    DatasetPaths paths;
    if (!ResolvePaths(config, &paths, error)) {
        return false;
    }
    int n = config.nbase;
    return ReadVectorsAutoF32(paths.base, &n, Xb, error);
}

bool LoadQuerySet(const DatasetConfig& config,
                  ColMajorMatrix<float>* Xq,
                  std::vector<int>* gt_first,
                  std::string* error) {
    if (!Xq || !gt_first) {
        if (error) {
            *error = "LoadQuerySet: output is null.";
        }
        return false;
    }
    DatasetPaths paths;
    if (!ResolvePaths(config, &paths, error)) {
        return false;
    }
    int n = config.nquery;
    if (!ReadVectorsAutoF32(paths.query, &n, Xq, error)) {
        return false;
    }
    int n_gt = n;
    const bool add_one = ShouldAddOne(config);
    if (!ReadGroundtruthFirst(paths.groundtruth, &n_gt, add_one, gt_first, error)) {
        return false;
    }
    if (n_gt != Xq->cols) {
        LogWarn("Ground truth count (" + std::to_string(n_gt) +
                ") differs from query vector count (" + std::to_string(Xq->cols) +
                "); shrinking Xq to match.");
        Xq->cols = n_gt;
        Xq->data.resize(static_cast<std::size_t>(Xq->rows) * static_cast<std::size_t>(n_gt));
    }
    return true;
}

bool ReadGroundtruthTopKFile(const std::string& path,
                             int nq,
                             int topk,
                             bool add_one,
                             ColMajorMatrix<int>* gt_topk,
                             std::string* error) {
    if (HasSuffix(path, ".ivecs")) {
        return ReadGroundtruthTopKIvecs(path, nq, topk, add_one, gt_topk, error);
    }
    if (HasSuffix(path, ".ibin")) {
        return ReadGroundtruthTopKIbin(path, nq, topk, add_one, gt_topk, error);
    }
    if (error) *error = "Unsupported groundtruth file extension for " + path;
    return false;
}

bool LoadGroundtruthTopK(const DatasetConfig& config,
                         int nq,
                         int topk,
                         ColMajorMatrix<int>* gt_topk,
                         std::string* error) {
    if (!gt_topk) {
        if (error) *error = "LoadGroundtruthTopK: output is null.";
        return false;
    }
    DatasetPaths paths;
    if (!ResolvePaths(config, &paths, error)) {
        return false;
    }
    return ReadGroundtruthTopKFile(paths.groundtruth, nq, topk, ShouldAddOne(config), gt_topk, error);
}

bool LoadExperimentData(const DatasetConfig& config, Dataset* dataset, std::string* error) {
    return LoadExperimentData(config, true, dataset, error);
}

bool LoadExperimentData(const DatasetConfig& config, bool load_train, Dataset* dataset, std::string* error) {
    if (!dataset) {
        if (error) {
            *error = "Dataset pointer is null.";
        }
        return false;
    }

    DatasetPaths paths;
    if (!ResolvePaths(config, &paths, error)) {
        return false;
    }

    if (load_train) {
        int n = config.ntrain;
        if (!ReadVectorsAutoF32(paths.train, &n, &dataset->Xt, error)) {
            return false;
        }
    } else {
        dataset->Xt = {};
    }
    {
        int n = config.nbase;
        if (!ReadVectorsAutoF32(paths.base, &n, &dataset->Xb, error)) {
            return false;
        }
    }
    {
        int n = config.nquery;
        if (!ReadVectorsAutoF32(paths.query, &n, &dataset->Xq, error)) {
            return false;
        }
        const bool add_one = ShouldAddOne(config);
        int n_gt = n;
        if (!ReadGroundtruthFirst(paths.groundtruth, &n_gt, add_one, &dataset->gt, error)) {
            return false;
        }
        if (n_gt != dataset->Xq.cols) {
            LogWarn("Ground truth count (" + std::to_string(n_gt) +
                    ") differs from query vector count (" + std::to_string(dataset->Xq.cols) +
                    "); shrinking Xq to match.");
            dataset->Xq.cols = n_gt;
            dataset->Xq.data.resize(static_cast<std::size_t>(dataset->Xq.rows) *
                                    static_cast<std::size_t>(n_gt));
        }
    }

    return true;
}

}  // namespace stlq::io
