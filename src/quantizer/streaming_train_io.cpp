#include "stlq/quantizer/streaming_train_io.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>

#include "stlq/io/dataset_reader_factory.h"
#include "stlq/common/logger.h"

namespace stlq {

namespace {

TrainFormat ToTrainFormat(io::VectorFileFormat format) {
    if (format == io::VectorFileFormat::kBvecs) return TrainFormat::kBvecs;
    if (format == io::VectorFileFormat::kFbin) return TrainFormat::kFbin;
    return TrainFormat::kFvecs;
}

class LimitColsReader final : public io::IColBlockReader, public io::IColBlockReaderTiming
#ifndef NDEBUG
    , public io::IColBlockReaderDebugStats
#endif
{
public:
    LimitColsReader(std::unique_ptr<io::IColBlockReader> base, std::int64_t limit)
        : base_(std::move(base)),
          limit_(std::max<std::int64_t>(0, limit)),
          next_(0) {}

    [[nodiscard]] int d() const override { return base_ ? base_->d() : 0; }
    [[nodiscard]] std::int64_t n() const override {
        if (!base_) return 0;
        return std::min<std::int64_t>(limit_, base_->n());
    }
    [[nodiscard]] io::ColBlockDType dtype() const override { return base_ ? base_->dtype() : io::ColBlockDType::kF32; }
    bool Reset(std::string* err) override {
        next_ = 0;
        return base_ ? base_->Reset(err) : false;
    }
    bool Seek(std::int64_t col0, std::string* err) override {
        const std::int64_t nn = n();
        next_ = std::min<std::int64_t>(std::max<std::int64_t>(0, col0), nn);
        if (!base_) return false;
        // Seek is best-effort; if base doesn't support it, fall back to Reset-at-0 semantics.
        if (base_->Seek(next_, err)) return true;
        if (next_ == 0) return base_->Reset(err);
        return false;
    }
    bool ReadNext(int max_cols, io::ColBlock* out, std::string* err) override {
        if (!out) {
            if (err) *err = "LimitColsReader::ReadNext: out is null.";
            return false;
        }
        const std::int64_t nn = n();
        if (next_ >= nn) {
            out->dtype = dtype();
            out->col0 = next_;
            out->X_f32 = {};
            out->X_u8 = {};
            return true;
        }
        const int want = static_cast<int>(std::min<std::int64_t>(std::max<std::int64_t>(1, max_cols), nn - next_));
        if (!base_->ReadNext(want, out, err)) return false;
        const int cols = (out->dtype == io::ColBlockDType::kU8) ? out->X_u8.cols : out->X_f32.cols;
        next_ = out->col0 + cols;
        return true;
    }
    bool ReadNextInto(int max_cols,
                      void* dst,
                      std::size_t dst_bytes,
                      std::int64_t* out_col0,
                      int* out_cols,
                      std::string* err) override {
        const std::int64_t nn = n();
        if (next_ >= nn) {
            if (out_col0) *out_col0 = next_;
            if (out_cols) *out_cols = 0;
            return true;
        }
        const int want = static_cast<int>(std::min<std::int64_t>(std::max<std::int64_t>(1, max_cols), nn - next_));
        const bool ok = base_->ReadNextInto(want, dst, dst_bytes, out_col0, out_cols, err);
        if (!ok) return false;
        if (out_col0 && out_cols) {
            next_ = (*out_col0) + (*out_cols);
        }
        return true;
    }

    [[nodiscard]] io::ColBlockReaderTimingBreakdown LastTimingBreakdown() const override {
        if (auto* t = dynamic_cast<io::IColBlockReaderTiming*>(base_.get())) {
            return t->LastTimingBreakdown();
        }
        return io::ColBlockReaderTimingBreakdown{};
    }

#ifndef NDEBUG
    io::ColBlockReaderDebugStats DebugStats() const override {
        if (auto* s = dynamic_cast<io::IColBlockReaderDebugStats*>(base_.get())) {
            return s->DebugStats();
        }
        return io::ColBlockReaderDebugStats{};
    }
#endif

private:
    std::unique_ptr<io::IColBlockReader> base_;
    std::int64_t limit_ = 0;
    std::int64_t next_ = 0;
};

}  // namespace

bool OpenTrainReaderAuto(const Config& cfg,
                         io::DatasetVectorReader* reader,
                         std::string* err) {
    if (!reader) {
        if (err) *err = "OpenTrainReaderAuto: reader output is null.";
        return false;
    }
    return io::OpenDatasetVectorReader(cfg, io::DatasetRole::kTrain, reader, err);
}

bool OpenStreamingTrainInput(const Config& config,
                             StreamingTrainInput* input,
                             std::string* error) {
    if (!input) {
        if (error) *error = "OpenStreamingTrainInput: input is null.";
        return false;
    }

    input->reader = {};
    if (!OpenTrainReaderAuto(config, &input->reader, error)) {
        return false;
    }
    input->train_format = ToTrainFormat(input->reader.format);
    input->train_is_u8 = (input->train_format == TrainFormat::kBvecs);

    input->d = input->reader.d;
    input->ntrain = input->reader.n;
    if (config.dataset.ntrain_set && config.dataset.ntrain > 0) {
        input->ntrain = std::min<std::uint64_t>(input->ntrain, static_cast<std::uint64_t>(config.dataset.ntrain));
    }
    if (input->d <= 0 || input->ntrain == 0) {
        if (error) *error = "TrainQuantizerStreamingLarge: empty train reader.";
        return false;
    }
    if (input->ntrain > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        if (error) *error = "TrainQuantizerStreamingLarge: ntrain exceeds int32 config limits.";
        return false;
    }
    return true;
}

std::unique_ptr<io::IColBlockReader> MakeTrainStreamingColBlockReader(const StreamingTrainInput& train_input,
                                                                      const KmeansConfig& kcfg,
                                                                      std::uint64_t ntrain,
                                                                      std::string* error) {
    std::string err;
    std::unique_ptr<io::IColBlockReader> base_reader;
    if (train_input.train_is_u8) {
        base_reader = kcfg.bvecs_use_u8 ? io::MakeBvecsColBlockReaderU8(train_input.reader.path, &err)
                                        : io::MakeBvecsColBlockReader(train_input.reader.path, &err);
    } else if (train_input.train_format == TrainFormat::kFbin) {
        base_reader = io::MakeFbinColBlockReader(train_input.reader.path, &err);
    } else {
        base_reader = io::MakeFvecsColBlockReader(train_input.reader.path, &err);
    }
    if (!base_reader) {
        if (error) *error = "TrainQuantizerStreamingLarge: failed to create train col-block reader: " + err;
        return nullptr;
    }

    // Wrap with LimitColsReader to restrict view to ntrain (virtual EOF).
    auto limit_reader = std::make_unique<LimitColsReader>(std::move(base_reader), static_cast<std::int64_t>(ntrain));

    // Optimization: Try to cache training data in Host RAM.
    // Now wrapping LimitColsReader: CachedReader will see EOF after ntrain, setting fully_loaded=true.
    std::string cache_err;
    auto reader = io::MakeCachedColBlockReader(std::move(limit_reader), &cache_err);
    if (!cache_err.empty()) {
        LogWarn("Train streaming init: Host RAM caching disabled (OOM or error): " + cache_err);
    } else {
        LogInfo("Train streaming init: Host RAM caching active.");
    }
    return reader;
}

}  // namespace stlq
