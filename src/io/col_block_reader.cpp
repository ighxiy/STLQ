#include "stlq/io/col_block_reader.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>

#include "stlq/io/bvecs_reader.h"
#include "stlq/io/fbin_reader.h"
#include "stlq/io/fvecs_reader.h"
#include "stlq/common/logger.h"

namespace stlq::io
{
    namespace
    {
        class MatrixColBlockReader final : public IColBlockReader
        {
        public:
            explicit MatrixColBlockReader(const ColMajorMatrix<float>& X) : X_(X)
            {
            }

            [[nodiscard]] int d() const override { return X_.rows; }
            [[nodiscard]] std::int64_t n() const override { return static_cast<std::int64_t>(X_.cols); }
            [[nodiscard]] ColBlockDType dtype() const override { return ColBlockDType::kF32; }

            bool Reset(std::string* /*err*/) override
            {
                next_ = 0;
                return true;
            }

            bool Seek(std::int64_t col0, std::string* /*err*/) override
            {
                const std::int64_t nn = n();
                next_ = std::min<std::int64_t>(std::max<std::int64_t>(0, col0), nn);
                return true;
            }

            bool ReadNext(int max_cols, ColBlock* out, std::string* err) override
            {
                if (!out)
                {
                    if (err) *err = "MatrixColBlockReader::ReadNext: out is null.";
                    return false;
                }
                const int d = X_.rows;
                const int n_i32 = X_.cols;
                const auto n64 = static_cast<std::int64_t>(n_i32);
                if (next_ >= n64)
                {
                    out->dtype = ColBlockDType::kF32;
                    out->X_f32.rows = 0;
                    out->X_f32.cols = 0;
                    out->X_f32.data.clear();
                    out->X_u8.rows = 0;
                    out->X_u8.cols = 0;
                    out->X_u8.data.clear();
                    out->col0 = n64;
                    return true;
                }
                const int cols = std::min(std::max(1, max_cols),
                                          static_cast<int>(std::min<std::int64_t>(n64 - next_, INT32_MAX)));
                out->dtype = ColBlockDType::kF32;
                out->X_f32.rows = d;
                out->X_f32.cols = cols;
                out->X_f32.data.resize(static_cast<std::size_t>(d) * static_cast<std::size_t>(cols));
                out->X_u8.rows = 0;
                out->X_u8.cols = 0;
                out->X_u8.data.clear();
                out->col0 = next_;
#pragma omp parallel for default(none) schedule(static) shared(out) firstprivate(cols, d)
                for (int j = 0; j < cols; ++j)
                {
                    const int src_col = static_cast<int>(next_) + j;
                    std::copy(X_.Col(src_col), X_.Col(src_col) + d, out->X_f32.Col(j));
                }
                next_ += cols;
                return true;
            }

            bool ReadNextInto(int max_cols,
                              void* dst,
                              std::size_t dst_bytes,
                              std::int64_t* out_col0,
                              int* out_cols,
                              std::string* err) override
            {
                if (!dst || !out_col0 || !out_cols)
                {
                    if (err) *err = "MatrixColBlockReader::ReadNextInto: null output.";
                    return false;
                }
                const int d = X_.rows;
                const std::int64_t n64 = n();
                if (next_ >= n64)
                {
                    *out_col0 = n64;
                    *out_cols = 0;
                    return true;
                }
                const int cols = std::min(std::max(1, max_cols),
                                          static_cast<int>(std::min<std::int64_t>(n64 - next_, INT32_MAX)));
                const std::size_t need = sizeof(float) * static_cast<std::size_t>(d) * static_cast<std::size_t>(cols);
                if (dst_bytes < need)
                {
                    if (err) *err = "MatrixColBlockReader::ReadNextInto: dst_bytes too small.";
                    return false;
                }
                auto* out = static_cast<float*>(dst);
#pragma omp parallel for default(none) schedule(static) shared(out) firstprivate(cols, d)
                for (int j = 0; j < cols; ++j)
                {
                    const int src_col = static_cast<int>(next_) + j;
                    std::memcpy(out + static_cast<std::size_t>(j) * static_cast<std::size_t>(d),
                                X_.Col(src_col),
                                sizeof(float) * static_cast<std::size_t>(d));
                }
                *out_col0 = next_;
                *out_cols = cols;
                next_ += cols;
                return true;
            }

        private:
            const ColMajorMatrix<float>& X_;
            std::int64_t next_ = 0;
        };

        class FvecsColBlockReader final : public IColBlockReader
        {
        public:
            explicit FvecsColBlockReader(FvecsReader reader) : reader_(std::move(reader))
            {
            }

            int d() const override { return reader_.d(); }
            std::int64_t n() const override { return static_cast<std::int64_t>(reader_.n()); }
            ColBlockDType dtype() const override { return ColBlockDType::kF32; }

            bool Reset(std::string* /*err*/) override
            {
                next_ = 0;
                return true;
            }

            bool Seek(std::int64_t col0, std::string* /*err*/) override
            {
                const std::int64_t nn = n();
                next_ = std::min<std::int64_t>(std::max<std::int64_t>(0, col0), nn);
                return true;
            }

            bool ReadNext(int max_cols, ColBlock* out, std::string* err) override
            {
                if (!out)
                {
                    if (err) *err = "FvecsColBlockReader::ReadNext: out is null.";
                    return false;
                }
                const std::int64_t n64 = n();
                if (next_ >= n64)
                {
                    out->dtype = ColBlockDType::kF32;
                    out->X_f32.rows = 0;
                    out->X_f32.cols = 0;
                    out->X_f32.data.clear();
                    out->X_u8.rows = 0;
                    out->X_u8.cols = 0;
                    out->X_u8.data.clear();
                    out->col0 = n64;
                    return true;
                }
                const int want = std::max(1, max_cols);
                const auto remaining = static_cast<std::uint64_t>(n64 - next_);
                const std::uint32_t count = static_cast<std::uint32_t>(std::min<std::uint64_t>(remaining, want));
                out->col0 = next_;
                out->dtype = ColBlockDType::kF32;
                out->X_u8.rows = 0;
                out->X_u8.cols = 0;
                out->X_u8.data.clear();
                if (!reader_.ReadBlock(static_cast<std::uint64_t>(next_), count, &out->X_f32, err))
                {
                    return false;
                }
                next_ += static_cast<std::int64_t>(count);
                return true;
            }

            bool ReadNextInto(int max_cols,
                              void* dst,
                              std::size_t dst_bytes,
                              std::int64_t* out_col0,
                              int* out_cols,
                              std::string* err) override
            {
                if (!dst || !out_col0 || !out_cols)
                {
                    if (err) *err = "FvecsColBlockReader::ReadNextInto: null output.";
                    return false;
                }
                const std::int64_t n64 = n();
                if (next_ >= n64)
                {
                    *out_col0 = n64;
                    *out_cols = 0;
                    return true;
                }
                const int want = std::max(1, max_cols);
                const auto remaining = static_cast<std::uint64_t>(n64 - next_);
                const std::uint32_t count = static_cast<std::uint32_t>(std::min<std::uint64_t>(remaining, want));
                if (!reader_.ReadBlockInto(static_cast<std::uint64_t>(next_),
                                           count,
                                           static_cast<float*>(dst),
                                           dst_bytes,
                                           err))
                {
                    return false;
                }
                *out_col0 = next_;
                *out_cols = static_cast<int>(count);
                next_ += static_cast<std::int64_t>(count);
                return true;
            }

        private:
            FvecsReader reader_;
            std::int64_t next_ = 0;
        };

        // FbinColBlockReader: identical to FvecsColBlockReader but wraps FbinReader (Yandex .fbin format).
        class FbinColBlockReader final : public IColBlockReader
        {
        public:
            explicit FbinColBlockReader(FbinReader reader) : reader_(std::move(reader))
            {
            }

            int d() const override { return reader_.d(); }
            std::int64_t n() const override { return static_cast<std::int64_t>(reader_.n()); }
            ColBlockDType dtype() const override { return ColBlockDType::kF32; }

            bool Reset(std::string* /*err*/) override
            {
                next_ = 0;
                return true;
            }

            bool Seek(std::int64_t col0, std::string* /*err*/) override
            {
                const std::int64_t nn = n();
                next_ = std::min<std::int64_t>(std::max<std::int64_t>(0, col0), nn);
                return true;
            }

            bool ReadNext(int max_cols, ColBlock* out, std::string* err) override
            {
                if (!out)
                {
                    if (err) *err = "FbinColBlockReader::ReadNext: out is null.";
                    return false;
                }
                const std::int64_t n64 = n();
                if (next_ >= n64)
                {
                    out->dtype = ColBlockDType::kF32;
                    out->X_f32.rows = 0;
                    out->X_f32.cols = 0;
                    out->X_f32.data.clear();
                    out->X_u8.rows = 0;
                    out->X_u8.cols = 0;
                    out->X_u8.data.clear();
                    out->col0 = n64;
                    return true;
                }
                const int want = std::max(1, max_cols);
                const auto remaining = static_cast<std::uint64_t>(n64 - next_);
                const std::uint32_t count = static_cast<std::uint32_t>(std::min<std::uint64_t>(remaining, want));
                out->col0 = next_;
                out->dtype = ColBlockDType::kF32;
                out->X_u8.rows = 0;
                out->X_u8.cols = 0;
                out->X_u8.data.clear();
                if (!reader_.ReadBlock(static_cast<std::uint64_t>(next_), count, &out->X_f32, err))
                {
                    return false;
                }
                next_ += static_cast<std::int64_t>(count);
                return true;
            }

            bool ReadNextInto(int max_cols,
                              void* dst,
                              std::size_t dst_bytes,
                              std::int64_t* out_col0,
                              int* out_cols,
                              std::string* err) override
            {
                if (!dst || !out_col0 || !out_cols)
                {
                    if (err) *err = "FbinColBlockReader::ReadNextInto: null output.";
                    return false;
                }
                const std::int64_t n64 = n();
                if (next_ >= n64)
                {
                    *out_col0 = n64;
                    *out_cols = 0;
                    return true;
                }
                const int want = std::max(1, max_cols);
                const auto remaining = static_cast<std::uint64_t>(n64 - next_);
                const std::uint32_t count = static_cast<std::uint32_t>(std::min<std::uint64_t>(remaining, want));
                if (!reader_.ReadBlockInto(static_cast<std::uint64_t>(next_),
                                           count,
                                           static_cast<float*>(dst),
                                           dst_bytes,
                                           err))
                {
                    return false;
                }
                *out_col0 = next_;
                *out_cols = static_cast<int>(count);
                next_ += static_cast<std::int64_t>(count);
                return true;
            }

        private:
            FbinReader reader_;
            std::int64_t next_ = 0;
        };

        class BvecsColBlockReader final : public IColBlockReader
#ifndef NDEBUG
    , public IColBlockReaderDebugStats
#endif
        {
        public:
            explicit BvecsColBlockReader(BvecsReader reader) : reader_(std::move(reader))
            {
            }

            int d() const override { return reader_.d(); }
            std::int64_t n() const override { return static_cast<std::int64_t>(reader_.n()); }
            ColBlockDType dtype() const override { return ColBlockDType::kF32; }

            bool Reset(std::string* /*err*/) override
            {
                next_ = 0;
                return true;
            }

            bool Seek(std::int64_t col0, std::string* /*err*/) override
            {
                const std::int64_t nn = n();
                next_ = std::min<std::int64_t>(std::max<std::int64_t>(0, col0), nn);
                return true;
            }

            bool ReadNext(int max_cols, ColBlock* out, std::string* err) override
            {
                if (!out)
                {
                    if (err) *err = "BvecsColBlockReader::ReadNext: out is null.";
                    return false;
                }
                const std::int64_t n64 = n();
                if (next_ >= n64)
                {
                    out->dtype = ColBlockDType::kF32;
                    out->X_f32.rows = 0;
                    out->X_f32.cols = 0;
                    out->X_f32.data.clear();
                    out->X_u8.rows = 0;
                    out->X_u8.cols = 0;
                    out->X_u8.data.clear();
                    out->col0 = n64;
                    return true;
                }
                const int want = std::max(1, max_cols);
                const auto remaining = static_cast<std::uint64_t>(n64 - next_);
                const std::uint32_t count = static_cast<std::uint32_t>(std::min<std::uint64_t>(remaining, want));
                out->col0 = next_;

                if (!reader_.ReadBlock(static_cast<std::uint64_t>(next_), count, &tmp_u8_, err))
                {
                    return false;
                }
                out->dtype = ColBlockDType::kF32;
                out->X_f32.rows = tmp_u8_.rows;
                out->X_f32.cols = tmp_u8_.cols;
                out->X_f32.data.resize(static_cast<std::size_t>(out->X_f32.rows) *
                    static_cast<std::size_t>(out->X_f32.cols));
                out->X_u8.rows = 0;
                out->X_u8.cols = 0;
                out->X_u8.data.clear();
                const int d = tmp_u8_.rows;
                const int cols = tmp_u8_.cols;
#pragma omp parallel for default(none) schedule(static) shared(out) firstprivate(cols, d)
                for (int j = 0; j < cols; ++j)
                {
                    const std::uint8_t* src = tmp_u8_.Col(j);
                    float* dst = out->X_f32.Col(j);
#pragma omp simd
                    for (int r = 0; r < d; ++r)
                    {
                        dst[r] = static_cast<float>(src[r]);
                    }
                }

                next_ += static_cast<std::int64_t>(count);
                return true;
            }

#ifndef NDEBUG
    ColBlockReaderDebugStats DebugStats() const override {
        ColBlockReaderDebugStats out;
        const auto st = reader_.debug_stats();
        out.read_calls = st.read_block_calls + st.read_block_into_calls;
        out.seek_calls = st.seek_calls;
        out.sequential_seek_skips = st.sequential_seek_skips;
        out.bytes_requested = st.bytes_requested;
        out.bytes_read = st.bytes_read;
        out.last_offset_bytes = st.last_offset_bytes;
        out.last_read_bytes = st.last_read_bytes;
        return out;
    }
#endif

        private:
            BvecsReader reader_;
            ColMajorMatrix<std::uint8_t> tmp_u8_;
            std::int64_t next_ = 0;
        };

        class BvecsColBlockReaderU8 final : public IColBlockReader, public IColBlockReaderTiming
#ifndef NDEBUG
    , public IColBlockReaderDebugStats
#endif
        {
        public:
            explicit BvecsColBlockReaderU8(BvecsReader reader) : reader_(std::move(reader))
            {
            }

            int d() const override { return reader_.d(); }
            std::int64_t n() const override { return static_cast<std::int64_t>(reader_.n()); }
            ColBlockDType dtype() const override { return ColBlockDType::kU8; }

            ColBlockReaderTimingBreakdown LastTimingBreakdown() const override
            {
                ColBlockReaderTimingBreakdown out;
                out.file_s = reader_.last_read_file_s();
                out.unpack_s = reader_.last_unpack_s();
                return out;
            }

#ifndef NDEBUG
    ColBlockReaderDebugStats DebugStats() const override {
        ColBlockReaderDebugStats out;
        const auto st = reader_.debug_stats();
        out.read_calls = st.read_block_calls + st.read_block_into_calls;
        out.seek_calls = st.seek_calls;
        out.sequential_seek_skips = st.sequential_seek_skips;
        out.bytes_requested = st.bytes_requested;
        out.bytes_read = st.bytes_read;
        out.last_offset_bytes = st.last_offset_bytes;
        out.last_read_bytes = st.last_read_bytes;
        return out;
    }
#endif

            bool Reset(std::string* /*err*/) override
            {
                next_ = 0;
                return true;
            }

            bool Seek(std::int64_t col0, std::string* /*err*/) override
            {
                const std::int64_t nn = n();
                next_ = std::min<std::int64_t>(std::max<std::int64_t>(0, col0), nn);
                return true;
            }

            bool ReadNext(int max_cols, ColBlock* out, std::string* err) override
            {
                if (!out)
                {
                    if (err) *err = "BvecsColBlockReaderU8::ReadNext: out is null.";
                    return false;
                }
                const std::int64_t n64 = n();
                if (next_ >= n64)
                {
                    out->dtype = ColBlockDType::kU8;
                    out->X_u8.rows = 0;
                    out->X_u8.cols = 0;
                    out->X_u8.data.clear();
                    out->X_f32.rows = 0;
                    out->X_f32.cols = 0;
                    out->X_f32.data.clear();
                    out->col0 = n64;
                    return true;
                }
                const int want = std::max(1, max_cols);
                const auto remaining = static_cast<std::uint64_t>(n64 - next_);
                const std::uint32_t count = static_cast<std::uint32_t>(std::min<std::uint64_t>(remaining, want));
                out->col0 = next_;

                out->dtype = ColBlockDType::kU8;
                out->X_f32.rows = 0;
                out->X_f32.cols = 0;
                out->X_f32.data.clear();
                if (!reader_.ReadBlock(static_cast<std::uint64_t>(next_), count, &out->X_u8, err))
                {
                    return false;
                }
                next_ += static_cast<std::int64_t>(count);
                return true;
            }

            bool ReadNextInto(int max_cols,
                              void* dst,
                              std::size_t dst_bytes,
                              std::int64_t* out_col0,
                              int* out_cols,
                              std::string* err) override
            {
                // NOTE: keep structure but drop hot null checks in release builds.
#ifndef NDEBUG
        if (!dst || !out_col0 || !out_cols) {
            if (err) *err = "BvecsColBlockReaderU8::ReadNextInto: null output.";
            return false;
        }
#endif
                const std::int64_t n64 = n();
                if (next_ >= n64)
                {
                    *out_col0 = n64;
                    *out_cols = 0;
                    return true;
                }
                const int want = std::max(1, max_cols);
                const auto remaining = static_cast<std::uint64_t>(n64 - next_);
                const std::uint32_t count = static_cast<std::uint32_t>(std::min<std::uint64_t>(remaining, want));
                if (!reader_.ReadBlockInto(static_cast<std::uint64_t>(next_),
                                           count,
                                           static_cast<std::uint8_t*>(dst),
                                           dst_bytes,
                                           err))
                {
                    return false;
                }
                *out_col0 = next_;
                *out_cols = static_cast<int>(count);
                next_ += static_cast<std::int64_t>(count);
                return true;
            }

        private:
            BvecsReader reader_;
            std::int64_t next_ = 0;
        };

        class CachedColBlockReader final : public IColBlockReader, public IColBlockReaderTiming
#ifndef NDEBUG
    , public IColBlockReaderDebugStats
#endif
        {
            enum class CacheState
            {
                kDisabled, // No cache or invalidated
                kBuilding, // Actively appending data
                kPartialValid, // Not appending, but holds valid prefix (e.g. after Seek)
                kFullyLoaded // All data cached
            };

        public:
            explicit CachedColBlockReader(std::unique_ptr<IColBlockReader> base) : base_(std::move(base))
            {
            }

            // Initialize the cache buffer. Returns false if allocation fails (OOM).
            bool InitCache(std::string* err)
            {
                if (!base_) return false;
                const int d = base_->d();
                const std::int64_t n = base_->n();
                std::size_t bytes_per_vec = 0;
                if (base_->dtype() == ColBlockDType::kU8)
                {
                    bytes_per_vec = sizeof(std::uint8_t) * static_cast<std::size_t>(d);
                }
                else
                {
                    bytes_per_vec = sizeof(float) * static_cast<std::size_t>(d);
                }
                const std::size_t total = bytes_per_vec * static_cast<std::size_t>(n);
                try
                {
                    // Allocate.
                    cache_.reset(new std::uint8_t[total]);
                    cache_cap_ = total;
                    cache_size_ = 0;
                    state_ = CacheState::kBuilding;
                    return true;
                }
                catch (...)
                {
                    if (err) *err = "CachedColBlockReader: OOM (failed to allocate " + std::to_string(total) +
                        " bytes)";
                    LogWarn(*err);
                    return false;
                }
            }

            [[nodiscard]] int d() const override { return base_->d(); }
            [[nodiscard]] std::int64_t n() const override { return base_->n(); }
            [[nodiscard]] ColBlockDType dtype() const override { return base_->dtype(); }

            [[nodiscard]] ColBlockReaderTimingBreakdown LastTimingBreakdown() const override
            {
                if (IsFullyLoaded())
                {
                    // Served from RAM.
                    return ColBlockReaderTimingBreakdown{0.0, last_memcpy_s_};
                }
                // Served from file (pass-through).
                if (auto* t = dynamic_cast<io::IColBlockReaderTiming*>(base_.get()))
                {
                    return t->LastTimingBreakdown();
                }
                return ColBlockReaderTimingBreakdown{};
            }

#ifndef NDEBUG
    ColBlockReaderDebugStats DebugStats() const override {
        if (auto* s = dynamic_cast<io::IColBlockReaderDebugStats*>(base_.get())) {
            return s->DebugStats();
        }
        return ColBlockReaderDebugStats{};
    }
#endif

            // Helper: Compute number of cached columns.
            [[nodiscard]] std::int64_t GetCachedCols() const
            {
                if (cache_size_ == 0) return 0;
                const int d = base_->d();
                const std::size_t bytes_per_col = (base_->dtype() == ColBlockDType::kU8)
                                                      ? (sizeof(std::uint8_t) * static_cast<std::size_t>(d))
                                                      : (sizeof(float) * static_cast<std::size_t>(d));
                return (bytes_per_col > 0) ? static_cast<std::int64_t>(cache_size_ / bytes_per_col) : 0;
            }

            // State helpers
            [[nodiscard]] bool IsFullyLoaded() const
            {
                // kFullyLoaded state or Building but accidentally hit cap
                return state_ == CacheState::kFullyLoaded ||
                    (state_ == CacheState::kBuilding && cache_cap_ > 0 && cache_size_ == cache_cap_);
            }

            [[nodiscard]] bool HasPartialCache() const
            {
                return state_ != CacheState::kDisabled && cache_size_ > 0;
            }

            [[nodiscard]] bool IsTargetInCache(std::int64_t target) const
            {
                if (state_ == CacheState::kDisabled) return false;
                if (IsFullyLoaded()) return true;
                const std::int64_t cache_cols = GetCachedCols();
                return cache_cols > 0 && target <= cache_cols;
            }

            void InvalidateCache()
            {
                state_ = CacheState::kDisabled;
                cache_size_ = 0;
            }

            void StartFreshCache()
            {
                if (cache_cap_ > 0 && cache_)
                {
                    state_ = CacheState::kBuilding;
                    cache_size_ = 0;
                }
                else
                {
                    state_ = CacheState::kDisabled;
                }
            }


            bool Reset(std::string* err) override
            {
                next_ = 0;

                if (IsFullyLoaded())
                {
                    return true;
                }

                if (HasPartialCache())
                {
                    // Preserve partial cache (important for mixed GPU cache mode).
                    return base_->Reset(err);
                }

                StartFreshCache();
                return base_->Reset(err);
            }

            bool Seek(std::int64_t col0, std::string* err) override
            {
                const std::int64_t target = std::min<std::int64_t>(std::max<std::int64_t>(0, col0), n());

                if (IsFullyLoaded())
                {
                    next_ = target;
                    return true;
                }

                // Preserve cache if seek target is within cached range (for mixed GPU cache).
                if (IsTargetInCache(target))
                {
                    next_ = target;
                    return base_->Seek(col0, err);
                }

                // Seek beyond cached range: invalidate cache.
                InvalidateCache();
                next_ = target;
                return base_->Seek(col0, err);
            }

            bool ReadNext(int max_cols, ColBlock* out, std::string* err) override
            {
                if (IsFullyLoaded())
                {
                    return ReadFromCache(max_cols, out, err);
                }
                // Read from base.
                if (!base_->ReadNext(max_cols, out, err)) return false;

                // Append to cache if active.
                if (state_ == CacheState::kBuilding)
                {
                    AppendToCache(*out);
                }
                if (out->X_f32.cols == 0 && out->X_u8.cols == 0)
                {
                    // EOF
                    if (state_ == CacheState::kBuilding) state_ = CacheState::kFullyLoaded;
                }
                else
                {
                    next_ += (out->dtype == ColBlockDType::kU8 ? out->X_u8.cols : out->X_f32.cols);
                }
                return true;
            }

            bool ReadNextInto(int max_cols,
                              void* dst,
                              std::size_t dst_bytes,
                              std::int64_t* out_col0,
                              int* out_cols,
                              std::string* err) override
            {
                if (IsFullyLoaded())
                {
                    return ReadFromCacheInto(max_cols, dst, dst_bytes, out_col0, out_cols, err);
                }

                // Direct read from base into dst.
                std::int64_t c0 = 0;
                int c = 0;
                // If base supports ReadNextInto, use it.
                // BUT we also need to copy that data into our cache.
                if (base_->ReadNextInto(max_cols, dst, dst_bytes, &c0, &c, err))
                {
                    if (c > 0 && state_ == CacheState::kBuilding)
                    {
                        AppendToCacheFromRaw(dst, c);
                    }
                    if (c == 0 && state_ == CacheState::kBuilding)
                    {
                        state_ = CacheState::kFullyLoaded;
                    }
                    if (out_col0) *out_col0 = c0;
                    if (out_cols) *out_cols = c;
                    next_ += c;
                    return true;
                }

                // Fallback: use ReadNext (ColBlock) -> copy to dst -> copy to cache.
                // Base reader likely failed because it doesn't implement ReadNextInto.
                // We can just call ReadNext (which populates cache via our override) then copy to dst.
                ColBlock blk;
                if (!ReadNext(max_cols, &blk, err)) return false;
                int cols = (blk.dtype == ColBlockDType::kU8) ? blk.X_u8.cols : blk.X_f32.cols;
                if (cols > 0)
                {
                    // Copy to dst.
                    const int d = (blk.dtype == ColBlockDType::kU8) ? blk.X_u8.rows : blk.X_f32.rows;
                    const std::size_t bytes_per = (blk.dtype == ColBlockDType::kU8)
                                                      ? sizeof(std::uint8_t)
                                                      : sizeof(float);
                    const std::size_t total = bytes_per * d * cols;
                    if (dst_bytes < total)
                    {
                        if (err) *err = "CachedColBlockReader::ReadNextInto: dst buffer too small.";
                        return false;
                    }
                    if (blk.dtype == ColBlockDType::kU8)
                    {
                        std::memcpy(dst, blk.X_u8.data.data(), total);
                    }
                    else
                    {
                        std::memcpy(dst, blk.X_f32.data.data(), total);
                    }
                }
                if (out_col0) *out_col0 = blk.col0;
                if (out_cols) *out_cols = cols;
                return true;
            }

        private:
            bool ReadFromCache(int max_cols, ColBlock* out, std::string* /*err*/)
            {
                const std::int64_t nn = n();
                const int cols = static_cast<int>(std::min<std::int64_t>(std::max(0, max_cols), nn - next_));
                out->col0 = next_;
                if (cols <= 0)
                {
                    out->X_f32.cols = 0;
                    out->X_u8.cols = 0;
                    out->dtype = dtype();
                    return true;
                }

                const int d = this->d();
                const auto dt = dtype();
                out->dtype = dt;

                const std::size_t bytes_per_vec = (dt == ColBlockDType::kU8)
                                                      ? sizeof(std::uint8_t) * d
                                                      : sizeof(float) * d;
                const std::size_t offset = static_cast<std::size_t>(next_) * bytes_per_vec;
                const std::uint8_t* src = cache_.get() + offset;

                using Clock = std::chrono::high_resolution_clock;
                auto t0 = Clock::now();

                if (dt == ColBlockDType::kU8)
                {
                    out->X_u8.rows = d;
                    out->X_u8.cols = cols;
                    out->X_u8.data.resize(static_cast<std::size_t>(d) * cols);
                    out->X_f32.rows = 0;
                    out->X_f32.cols = 0;
                    std::memcpy(out->X_u8.data.data(), src, bytes_per_vec * cols);
                }
                else
                {
                    out->X_f32.rows = d;
                    out->X_f32.cols = cols;
                    out->X_f32.data.resize(static_cast<std::size_t>(d) * cols);
                    out->X_u8.rows = 0;
                    out->X_u8.cols = 0;
                    std::memcpy(out->X_f32.data.data(), src, bytes_per_vec * cols);
                }
                auto t1 = Clock::now();
                last_memcpy_s_ = std::chrono::duration<double>(t1 - t0).count();

                next_ += cols;
                return true;
            }

            bool ReadFromCacheInto(int max_cols,
                                   void* dst,
                                   std::size_t dst_bytes,
                                   std::int64_t* out_col0,
                                   int* out_cols,
                                   std::string* err)
            {
                const std::int64_t nn = n();
                const int cols = static_cast<int>(std::min<std::int64_t>(std::max(0, max_cols), nn - next_));
                if (out_col0) *out_col0 = next_;
                if (out_cols) *out_cols = cols;
                if (cols <= 0) return true;

                const int d = this->d();
                const auto dt = dtype();
                const std::size_t bytes_per_vec = (dt == ColBlockDType::kU8)
                                                      ? sizeof(std::uint8_t) * d
                                                      : sizeof(float) * d;
                const std::size_t total = bytes_per_vec * cols;
                if (dst_bytes < total)
                {
                    if (err) *err = "CachedColBlockReader::ReadFromCacheInto: buffer too small.";
                    return false;
                }

                const std::size_t offset = static_cast<std::size_t>(next_) * bytes_per_vec;
                const std::uint8_t* src = cache_.get() + offset;

                using Clock = std::chrono::high_resolution_clock;
                auto t0 = Clock::now();
                std::memcpy(dst, src, total);
                auto t1 = Clock::now();
                last_memcpy_s_ = std::chrono::duration<double>(t1 - t0).count();

                next_ += cols;
                return true;
            }

            void AppendToCache(const ColBlock& blk)
            {
                if (state_ != CacheState::kBuilding) return;

                const int cols = (blk.dtype == ColBlockDType::kU8) ? blk.X_u8.cols : blk.X_f32.cols;
                if (cols <= 0) return;
                const int d = (blk.dtype == ColBlockDType::kU8) ? blk.X_u8.rows : blk.X_f32.rows;
                const std::size_t vec_bytes = (blk.dtype == ColBlockDType::kU8)
                                                  ? (sizeof(std::uint8_t) * d)
                                                  : (sizeof(float) * d);
                const std::size_t total_bytes = vec_bytes * cols;

                if (cache_size_ + total_bytes > cache_cap_)
                {
                    state_ = CacheState::kDisabled; // OOM or logic error
                    cache_size_ = 0;
                    return;
                }

                std::uint8_t* dst_base = cache_.get() + cache_size_;
                if (blk.dtype == ColBlockDType::kU8)
                {
                    const std::uint8_t* src_base = blk.X_u8.data.data();
#pragma omp parallel for default(none) schedule(static) shared(dst_base, src_base) firstprivate(cols, vec_bytes)
                    for (int c = 0; c < cols; ++c)
                    {
                        std::memcpy(dst_base + c * vec_bytes, src_base + c * vec_bytes, vec_bytes);
                    }
                }
                else
                {
                    const auto* src_base = reinterpret_cast<const std::uint8_t*>(blk.X_f32.data.data());
#pragma omp parallel for default(none) schedule(static) shared(dst_base, src_base) firstprivate(cols, vec_bytes)
                    for (int c = 0; c < cols; ++c)
                    {
                        std::memcpy(dst_base + c * vec_bytes, src_base + c * vec_bytes, vec_bytes);
                    }
                }
                cache_size_ += total_bytes;
            }

            void AppendToCacheFromRaw(const void* src, int cols)
            {
                if (state_ != CacheState::kBuilding) return;

                const int d = this->d();
                const auto dt = dtype();
                const std::size_t vec_bytes = (dt == ColBlockDType::kU8)
                                                  ? (sizeof(std::uint8_t) * d)
                                                  : (sizeof(float) * d);
                const std::size_t total_bytes = vec_bytes * cols;

                if (cache_size_ + total_bytes > cache_cap_)
                {
                    state_ = CacheState::kDisabled;
                    cache_size_ = 0;
                    return;
                }

                std::uint8_t* dst_base = cache_.get() + cache_size_;
                const auto* src_u8 = static_cast<const std::uint8_t*>(src);

#pragma omp parallel for default(none) schedule(static) shared(dst_base, src_u8) firstprivate(cols, vec_bytes)
                for (int c = 0; c < cols; ++c)
                {
                    std::memcpy(dst_base + c * vec_bytes, src_u8 + c * vec_bytes, vec_bytes);
                }
                cache_size_ += total_bytes;
            }

            std::unique_ptr<io::IColBlockReader> base_;
            std::unique_ptr<std::uint8_t[]> cache_;
            std::size_t cache_cap_ = 0;
            std::size_t cache_size_ = 0;
            CacheState state_ = CacheState::kDisabled;
            std::int64_t next_ = 0;
            double last_memcpy_s_ = 0.0;
        };
    } // namespace

    std::unique_ptr<IColBlockReader> MakeMatrixColBlockReader(const ColMajorMatrix<float>& X)
    {
        return std::make_unique<MatrixColBlockReader>(X);
    }

    std::unique_ptr<IColBlockReader> MakeFvecsColBlockReader(const std::string& path,
                                                             std::string* err)
    {
        FvecsReader reader;
        if (!reader.Open(path, err))
        {
            return nullptr;
        }
        return std::make_unique<FvecsColBlockReader>(std::move(reader));
    }

    std::unique_ptr<IColBlockReader> MakeFbinColBlockReader(const std::string& path,
                                                            std::string* err)
    {
        FbinReader reader;
        if (!reader.Open(path, err))
        {
            return nullptr;
        }
        return std::make_unique<FbinColBlockReader>(std::move(reader));
    }

    std::unique_ptr<IColBlockReader> MakeBvecsColBlockReader(const std::string& path,
                                                             std::string* err)
    {
        BvecsReader reader;
        if (!reader.Open(path, err))
        {
            return nullptr;
        }
        return std::make_unique<BvecsColBlockReader>(std::move(reader));
    }

    std::unique_ptr<IColBlockReader> MakeBvecsColBlockReaderU8(const std::string& path,
                                                               std::string* err)
    {
        BvecsReader reader;
        if (!reader.Open(path, err))
        {
            return nullptr;
        }
        return std::make_unique<BvecsColBlockReaderU8>(std::move(reader));
    }

    std::unique_ptr<IColBlockReader> MakeCachedColBlockReader(std::unique_ptr<IColBlockReader> base,
                                                              std::string* err)
    {
        auto wrapper = std::make_unique<CachedColBlockReader>(std::move(base));
        // Try to allocate cache. If fail, log/error, but here we can just return the wrapper
        // which will have caching_active_=false (InitCache false -> false).
        // Wait, wrapper ctor doesn't call InitCache.
        if (!wrapper->InitCache(err))
        {
            // Fallback: return the original base?
            // No, CachedColBlockReader works with caching_active_=false as pass-through.
            // BUT it's heavy to have a wrapper that does nothing.
            // Let's propagate the error if caller cares, or print warning.
            // The prompt says "automatic fallback".
            // If InitCache failed, wrapper is just a pass-through timing wrapper properly (LastTimingBreakdown will call base).
            // Let's just return it. The user will see "OOM" in error string maybe?
            // Let's rely on caller to notice efficiency or log warnings.
            // Actually, if InitCache fails, `err` will contain "OOM...".
            // The implementation above handles !caching_active_ correctly.
        }
        return wrapper;
    }
} // namespace stlq::io
