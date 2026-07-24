#include "stlq/io/dataset_reader_factory.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <string>

#include "stlq/io/dataset_io.h"
#include "stlq/common/logger.h"

namespace stlq::io {
namespace {

std::string ToLower(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

bool HasSuffixInsensitive(const std::string& s, const char* suffix) {
    const std::string a = ToLower(s);
    const std::string b = ToLower(suffix);
    return a.size() >= b.size() && a.compare(a.size() - b.size(), b.size(), b) == 0;
}

std::string RolePathHint(const Config& config, DatasetRole role) {
    switch (role) {
    case DatasetRole::kTrain:
        return config.dataset.train_path;
    case DatasetRole::kBase:
        return config.dataset.base_path;
    case DatasetRole::kQuery:
        return config.dataset.query_path;
    }
    return {};
}

std::string RoleFormatOverride(const Config& config, DatasetRole role) {
    switch (role) {
    case DatasetRole::kTrain:
        return config.large.train_format;
    case DatasetRole::kBase:
        return config.large.base_format;
    case DatasetRole::kQuery:
        return config.large.query_format;
    }
    return "auto";
}

VectorFileFormat ParseVectorFormat(const std::string& raw) {
    const std::string s = ToLower(raw);
    if (s.empty() || s == "auto") return VectorFileFormat::kAuto;
    if (s == "bvecs" || s == "siftbin" || s == "u8") return VectorFileFormat::kBvecs;
    if (s == "fvecs" || s == "float" || s == "f32") return VectorFileFormat::kFvecs;
    if (s == "fbin") return VectorFileFormat::kFbin;
    return VectorFileFormat::kAuto;
}

bool IsVectorFormatTokenValid(const std::string& raw) {
    const std::string s = ToLower(raw);
    return s.empty() || s == "auto" ||
           s == "bvecs" || s == "siftbin" || s == "u8" ||
           s == "fvecs" || s == "float" || s == "f32" ||
           s == "fbin";
}

GroundtruthFileFormat ParseGroundtruthFormat(const std::string& raw) {
    const std::string s = ToLower(raw);
    if (s.empty() || s == "auto") return GroundtruthFileFormat::kAuto;
    if (s == "ivecs") return GroundtruthFileFormat::kIvecs;
    if (s == "ibin") return GroundtruthFileFormat::kIbin;
    return GroundtruthFileFormat::kAuto;
}

bool IsGroundtruthFormatTokenValid(const std::string& raw) {
    const std::string s = ToLower(raw);
    return s.empty() || s == "auto" || s == "ivecs" || s == "ibin";
}

VectorFileFormat FormatFromPathHint(const std::string& path) {
    if (HasSuffixInsensitive(path, ".bvecs") || HasSuffixInsensitive(path, ".siftbin")) {
        return VectorFileFormat::kBvecs;
    }
    if (HasSuffixInsensitive(path, ".fbin")) {
        return VectorFileFormat::kFbin;
    }
    if (HasSuffixInsensitive(path, ".fvecs")) {
        return VectorFileFormat::kFvecs;
    }
    return VectorFileFormat::kAuto;
}

GroundtruthFileFormat GroundtruthFormatFromPathHint(const std::string& path) {
    if (HasSuffixInsensitive(path, ".ibin")) return GroundtruthFileFormat::kIbin;
    if (HasSuffixInsensitive(path, ".ivecs")) return GroundtruthFileFormat::kIvecs;
    return GroundtruthFileFormat::kAuto;
}

void SetBvecs(DatasetVectorReader* out, BvecsReader&& reader) {
    out->format = VectorFileFormat::kBvecs;
    out->is_u8 = true;
    out->d = reader.d();
    out->n = reader.n();
    out->path = reader.path();
    out->bvecs = std::move(reader);
}

void SetFvecs(DatasetVectorReader* out, FvecsReader&& reader) {
    out->format = VectorFileFormat::kFvecs;
    out->is_u8 = false;
    out->d = reader.d();
    out->n = reader.n();
    out->path = reader.path();
    out->fvecs = std::move(reader);
}

void SetFbin(DatasetVectorReader* out, FbinReader&& reader) {
    out->format = VectorFileFormat::kFbin;
    out->is_u8 = false;
    out->d = reader.d();
    out->n = reader.n();
    out->path = reader.path();
    out->fbin = std::move(reader);
}

bool TryOpenVectorFormat(const Config& config,
                         DatasetRole role,
                         VectorFileFormat format,
                         DatasetVectorReader* out,
                         std::string* error) {
    std::string local_err;
    switch (format) {
    case VectorFileFormat::kBvecs: {
        BvecsReader reader;
        bool ok = false;
        if (role == DatasetRole::kTrain) {
            ok = OpenTrainBvecsReader(config, &reader, &local_err);
        } else if (role == DatasetRole::kBase) {
            ok = OpenBaseBvecsReader(config, &reader, &local_err);
        } else {
            ok = OpenQueryBvecsReader(config, &reader, &local_err);
        }
        if (ok) {
            SetBvecs(out, std::move(reader));
            return true;
        }
        break;
    }
    case VectorFileFormat::kFvecs: {
        FvecsReader reader;
        bool ok = false;
        if (role == DatasetRole::kTrain) {
            ok = OpenTrainFvecsReader(config, &reader, &local_err);
        } else if (role == DatasetRole::kBase) {
            ok = OpenBaseFvecsReader(config, &reader, &local_err);
        } else {
            ok = OpenQueryFvecsReader(config, &reader, &local_err);
        }
        if (ok) {
            SetFvecs(out, std::move(reader));
            return true;
        }
        break;
    }
    case VectorFileFormat::kFbin: {
        FbinReader reader;
        bool ok = false;
        if (role == DatasetRole::kTrain) {
            ok = OpenTrainFbinReader(config, &reader, &local_err);
        } else if (role == DatasetRole::kBase) {
            ok = OpenBaseFbinReader(config, &reader, &local_err);
        } else {
            ok = OpenQueryFbinReader(config, &reader, &local_err);
        }
        if (ok) {
            SetFbin(out, std::move(reader));
            return true;
        }
        break;
    }
    case VectorFileFormat::kAuto:
        break;
    }
    if (error && error->empty()) {
        *error = local_err;
    }
    return false;
}

std::vector<VectorFileFormat> AutoVectorPriority(const Config& config, DatasetRole role) {
    const VectorFileFormat suffix_fmt = FormatFromPathHint(RolePathHint(config, role));
    if (suffix_fmt != VectorFileFormat::kAuto) {
        std::vector<VectorFileFormat> order{suffix_fmt};
        for (VectorFileFormat f : {VectorFileFormat::kFvecs, VectorFileFormat::kFbin, VectorFileFormat::kBvecs}) {
            if (std::find(order.begin(), order.end(), f) == order.end()) order.push_back(f);
        }
        return order;
    }
    if (config.dataset.name == "SIFT1B") {
        return {VectorFileFormat::kBvecs, VectorFileFormat::kFvecs, VectorFileFormat::kFbin};
    }
    if (config.dataset.name == "DEEP10M") {
        return {VectorFileFormat::kFbin, VectorFileFormat::kFvecs, VectorFileFormat::kBvecs};
    }
    return {VectorFileFormat::kFvecs, VectorFileFormat::kFbin, VectorFileFormat::kBvecs};
}

bool ApplyAddOne(std::int32_t raw, bool add_one, int* out) {
    if (!out) return false;
    if (add_one) raw += 1;
    *out = static_cast<int>(raw);
    return true;
}

}  // namespace

const char* DatasetRoleName(DatasetRole role) {
    switch (role) {
    case DatasetRole::kTrain:
        return "train";
    case DatasetRole::kBase:
        return "base";
    case DatasetRole::kQuery:
        return "query";
    }
    return "unknown";
}

const char* VectorFileFormatName(VectorFileFormat format) {
    switch (format) {
    case VectorFileFormat::kAuto:
        return "auto";
    case VectorFileFormat::kBvecs:
        return "bvecs";
    case VectorFileFormat::kFvecs:
        return "fvecs";
    case VectorFileFormat::kFbin:
        return "fbin";
    }
    return "unknown";
}

const char* GroundtruthFileFormatName(GroundtruthFileFormat format) {
    switch (format) {
    case GroundtruthFileFormat::kAuto:
        return "auto";
    case GroundtruthFileFormat::kIvecs:
        return "ivecs";
    case GroundtruthFileFormat::kIbin:
        return "ibin";
    }
    return "unknown";
}

bool OpenDatasetVectorReader(const Config& config,
                             DatasetRole role,
                             DatasetVectorReader* out,
                             std::string* error) {
    if (!out) {
        if (error) *error = "OpenDatasetVectorReader: output is null.";
        return false;
    }
    if (error) error->clear();
    *out = DatasetVectorReader{};

    const std::string requested_raw = RoleFormatOverride(config, role);
    if (!IsVectorFormatTokenValid(requested_raw)) {
        if (error) {
            *error = std::string("OpenDatasetVectorReader(") + DatasetRoleName(role) +
                     "): unsupported format override \"" + requested_raw + "\".";
        }
        return false;
    }
    const VectorFileFormat requested = ParseVectorFormat(requested_raw);
    if (requested != VectorFileFormat::kAuto) {
        std::string local_err;
        if (TryOpenVectorFormat(config, role, requested, out, &local_err)) {
            return true;
        }
        if (error) {
            *error = std::string("OpenDatasetVectorReader(") + DatasetRoleName(role) +
                     "): failed to open requested format " + VectorFileFormatName(requested) +
                     (local_err.empty() ? "." : ": " + local_err);
        }
        return false;
    }

    std::string first_err;
    for (VectorFileFormat fmt : AutoVectorPriority(config, role)) {
        std::string local_err;
        if (TryOpenVectorFormat(config, role, fmt, out, &local_err)) {
            return true;
        }
        if (first_err.empty()) first_err = local_err;
    }
    if (error) {
        *error = std::string("OpenDatasetVectorReader(") + DatasetRoleName(role) +
                 "): failed to open vector file as fvecs/fbin/bvecs" +
                 (first_err.empty() ? "." : ": " + first_err);
    }
    return false;
}

bool ReadDatasetVectorBlockF32(const DatasetVectorReader& reader,
                               std::uint64_t start,
                               std::uint32_t count,
                               ColMajorMatrix<float>* out,
                               std::string* error) {
    if (!out) {
        if (error) *error = "ReadDatasetVectorBlockF32: output is null.";
        return false;
    }
    *out = {};
    if (count == 0) return true;
    if (reader.d <= 0) {
        if (error) *error = "ReadDatasetVectorBlockF32: invalid reader dimension.";
        return false;
    }
    if (reader.format == VectorFileFormat::kBvecs) {
        ColMajorMatrix<std::uint8_t> tmp_u8;
        if (!reader.bvecs.ReadBlock(start, count, &tmp_u8, error)) return false;
        ConvertBvecsToF32(tmp_u8, out);
        return true;
    }
    if (reader.format == VectorFileFormat::kFvecs) {
        return reader.fvecs.ReadBlock(start, count, out, error);
    }
    if (reader.format == VectorFileFormat::kFbin) {
        return reader.fbin.ReadBlock(start, count, out, error);
    }
    if (error) *error = "ReadDatasetVectorBlockF32: no concrete reader.";
    return false;
}

bool ReadDatasetVectorBlockU8(const DatasetVectorReader& reader,
                              std::uint64_t start,
                              std::uint32_t count,
                              ColMajorMatrix<std::uint8_t>* out,
                              std::string* error) {
    if (!out) {
        if (error) *error = "ReadDatasetVectorBlockU8: output is null.";
        return false;
    }
    *out = {};
    if (count == 0) return true;
    if (reader.d <= 0) {
        if (error) *error = "ReadDatasetVectorBlockU8: invalid reader dimension.";
        return false;
    }
    if (reader.format != VectorFileFormat::kBvecs) {
        if (error) *error = "ReadDatasetVectorBlockU8: reader is not u8.";
        return false;
    }
    return reader.bvecs.ReadBlock(start, count, out, error);
}

bool ReadDatasetVectorByIdsF32(const DatasetVectorReader& reader,
                               const std::vector<std::uint32_t>& ids,
                               ColMajorMatrix<float>* out,
                               std::string* error) {
    if (!out) {
        if (error) *error = "ReadDatasetVectorByIdsF32: output is null.";
        return false;
    }
    *out = {};
    if (ids.empty()) return true;
    if (reader.d <= 0) {
        if (error) *error = "ReadDatasetVectorByIdsF32: invalid reader dimension.";
        return false;
    }
    if (reader.format == VectorFileFormat::kBvecs) {
        ColMajorMatrix<std::uint8_t> tmp_u8;
        if (!reader.bvecs.ReadByIds(ids, &tmp_u8, error)) return false;
        ConvertBvecsToF32(tmp_u8, out);
        return true;
    }
    if (reader.format == VectorFileFormat::kFvecs) {
        return reader.fvecs.ReadByIds(ids, out, error);
    }
    if (reader.format == VectorFileFormat::kFbin) {
        return reader.fbin.ReadByIds(ids, out, error);
    }
    if (error) *error = "ReadDatasetVectorByIdsF32: no concrete reader.";
    return false;
}

bool ReadDatasetVectorByIdsU8(const DatasetVectorReader& reader,
                              const std::vector<std::uint32_t>& ids,
                              ColMajorMatrix<std::uint8_t>* out,
                              std::string* error) {
    if (!out) {
        if (error) *error = "ReadDatasetVectorByIdsU8: output is null.";
        return false;
    }
    *out = {};
    if (ids.empty()) return true;
    if (reader.d <= 0) {
        if (error) *error = "ReadDatasetVectorByIdsU8: invalid reader dimension.";
        return false;
    }
    if (reader.format != VectorFileFormat::kBvecs) {
        if (error) *error = "ReadDatasetVectorByIdsU8: reader is not u8.";
        return false;
    }
    return reader.bvecs.ReadByIds(ids, out, error);
}

bool OpenDatasetGroundtruthReader(const Config& config,
                                  DatasetGroundtruthReader* out,
                                  std::string* error) {
    if (!out) {
        if (error) *error = "OpenDatasetGroundtruthReader: output is null.";
        return false;
    }
    if (error) error->clear();
    *out = DatasetGroundtruthReader{};

    if (!IsGroundtruthFormatTokenValid(config.large.gt_format)) {
        if (error) {
            *error = "OpenDatasetGroundtruthReader: unsupported format override \"" +
                     config.large.gt_format + "\".";
        }
        return false;
    }
    GroundtruthFileFormat requested = ParseGroundtruthFormat(config.large.gt_format);
    if (requested == GroundtruthFileFormat::kAuto) {
        requested = GroundtruthFormatFromPathHint(config.dataset.groundtruth_path);
    }

    auto try_ivecs = [&]() -> bool {
        IvecsReader reader;
        std::string local_err;
        if (!OpenGroundtruthIvecsReader(config, &reader, &local_err)) {
            if (error && error->empty()) *error = local_err;
            return false;
        }
        out->format = GroundtruthFileFormat::kIvecs;
        out->k = reader.k();
        out->n = reader.n();
        out->path = reader.path();
        out->ivecs = std::move(reader);
        return true;
    };
    auto try_ibin = [&]() -> bool {
        IbinReader reader;
        std::string local_err;
        if (!OpenGroundtruthIbinReader(config, &reader, &local_err)) {
            if (error && error->empty()) *error = local_err;
            return false;
        }
        out->format = GroundtruthFileFormat::kIbin;
        out->k = reader.k();
        out->n = reader.n();
        out->path = reader.path();
        out->ibin = std::move(reader);
        return true;
    };

    if (requested == GroundtruthFileFormat::kIvecs) {
        if (try_ivecs()) return true;
        if (error && !error->empty()) *error = "OpenDatasetGroundtruthReader: failed requested ivecs: " + *error;
        return false;
    }
    if (requested == GroundtruthFileFormat::kIbin) {
        if (try_ibin()) return true;
        if (error && !error->empty()) *error = "OpenDatasetGroundtruthReader: failed requested ibin: " + *error;
        return false;
    }
    if (config.dataset.name == "DEEP10M") {
        if (try_ibin()) return true;
        if (error) error->clear();
        if (try_ivecs()) return true;
    } else {
        if (try_ivecs()) return true;
        if (error) error->clear();
        if (try_ibin()) return true;
    }
    if (error) *error = "OpenDatasetGroundtruthReader: failed to open groundtruth as ivecs/ibin.";
    return false;
}

bool ReadDatasetGroundtruthFirst(const Config& config,
                                 const DatasetGroundtruthReader& reader,
                                 int* n_io,
                                 std::vector<int>* out,
                                 std::string* error) {
    if (!n_io || *n_io <= 0 || !out) {
        if (error) *error = "ReadDatasetGroundtruthFirst: invalid args.";
        return false;
    }
    if (reader.k <= 0 || reader.n == 0) {
        if (error) *error = "ReadDatasetGroundtruthFirst: empty groundtruth reader.";
        return false;
    }
    const int available = static_cast<int>(
        std::min<std::uint64_t>(reader.n, static_cast<std::uint64_t>(std::numeric_limits<int>::max())));
    int n = std::min(*n_io, available);
    if (*n_io > available) {
        LogWarn("ReadDatasetGroundtruthFirst: requested " + std::to_string(*n_io) +
                " but file has " + std::to_string(reader.n) + "; shrinking to " +
                std::to_string(n) + ".");
    }
    *n_io = n;

    const bool add_one = config.dataset.groundtruth_add1 != 0;
    out->assign(static_cast<std::size_t>(n), 0);
    if (reader.format == GroundtruthFileFormat::kIvecs) {
        ColMajorMatrix<std::int32_t> raw;
        if (!reader.ivecs.ReadBlock(0, static_cast<std::uint32_t>(n), &raw, error)) return false;
        for (int i = 0; i < n; ++i) {
            ApplyAddOne(raw.Col(i)[0], add_one, &(*out)[static_cast<std::size_t>(i)]);
        }
        return true;
    }
    if (reader.format == GroundtruthFileFormat::kIbin) {
        ColMajorMatrix<std::int32_t> raw;
        if (!reader.ibin.ReadBlock(0, static_cast<std::uint32_t>(n), &raw, error)) return false;
        for (int i = 0; i < n; ++i) {
            ApplyAddOne(raw.Col(i)[0], add_one, &(*out)[static_cast<std::size_t>(i)]);
        }
        return true;
    }
    if (error) *error = "ReadDatasetGroundtruthFirst: no concrete reader.";
    return false;
}

bool ReadDatasetGroundtruthTopK(const Config& config,
                                const DatasetGroundtruthReader& reader,
                                int nq,
                                int topk,
                                ColMajorMatrix<int>* out,
                                std::string* error) {
    if (!out || nq <= 0 || topk <= 0) {
        if (error) *error = "ReadDatasetGroundtruthTopK: invalid args.";
        return false;
    }
    if (reader.k <= 0 || reader.n == 0) {
        if (error) *error = "ReadDatasetGroundtruthTopK: empty groundtruth reader.";
        return false;
    }
    const int nq_file = static_cast<int>(
        std::min<std::uint64_t>(reader.n, static_cast<std::uint64_t>(std::numeric_limits<int>::max())));
    if (nq > nq_file) {
        if (error) *error = "ReadDatasetGroundtruthTopK: requested nq=" + std::to_string(nq) +
                            " exceeds file rows=" + std::to_string(nq_file);
        return false;
    }
    const int out_k = std::min(topk, reader.k);
    const bool add_one = config.dataset.groundtruth_add1 != 0;
    ColMajorMatrix<std::int32_t> raw;
    if (reader.format == GroundtruthFileFormat::kIvecs) {
        if (!reader.ivecs.ReadBlock(0, static_cast<std::uint32_t>(nq), &raw, error)) return false;
    } else if (reader.format == GroundtruthFileFormat::kIbin) {
        if (!reader.ibin.ReadBlock(0, static_cast<std::uint32_t>(nq), &raw, error)) return false;
    } else {
        if (error) *error = "ReadDatasetGroundtruthTopK: no concrete reader.";
        return false;
    }

    out->rows = out_k;
    out->cols = nq;
    out->data.resize(static_cast<std::size_t>(out_k) * static_cast<std::size_t>(nq));
    for (int q = 0; q < nq; ++q) {
        const std::int32_t* src = raw.Col(q);
        int* dst = out->Col(q);
        for (int i = 0; i < out_k; ++i) {
            ApplyAddOne(src[i], add_one, &dst[i]);
        }
    }
    return true;
}

bool LoadQuerySet(const Config& config,
                  ColMajorMatrix<float>* Xq,
                  std::vector<int>* gt_first,
                  std::string* error) {
    if (!Xq || !gt_first) {
        if (error) *error = "LoadQuerySet: output is null.";
        return false;
    }
    DatasetVectorReader query_reader;
    if (!OpenDatasetVectorReader(config, DatasetRole::kQuery, &query_reader, error)) {
        return false;
    }
    std::uint64_t nq_u64 = query_reader.n;
    if (config.dataset.nquery_set && config.dataset.nquery > 0) {
        nq_u64 = std::min<std::uint64_t>(nq_u64, static_cast<std::uint64_t>(config.dataset.nquery));
    }
    const std::uint32_t nq = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(nq_u64, static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())));
    if (!ReadDatasetVectorBlockF32(query_reader, 0, nq, Xq, error)) {
        return false;
    }

    DatasetGroundtruthReader gt_reader;
    if (!OpenDatasetGroundtruthReader(config, &gt_reader, error)) {
        return false;
    }
    int n_gt = static_cast<int>(std::min<std::uint64_t>(nq, static_cast<std::uint64_t>(std::numeric_limits<int>::max())));
    if (!ReadDatasetGroundtruthFirst(config, gt_reader, &n_gt, gt_first, error)) {
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

bool LoadGroundtruthTopK(const Config& config,
                         int nq,
                         int topk,
                         ColMajorMatrix<int>* gt_topk,
                         std::string* error) {
    DatasetGroundtruthReader reader;
    if (!OpenDatasetGroundtruthReader(config, &reader, error)) {
        return false;
    }
    return ReadDatasetGroundtruthTopK(config, reader, nq, topk, gt_topk, error);
}

}  // namespace stlq::io
