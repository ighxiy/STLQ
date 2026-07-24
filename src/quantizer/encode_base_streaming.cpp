#include "stlq/quantizer/encode_base_streaming.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <condition_variable>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "stlq/core/blas.h"
#include "stlq/core/kernel_provider_cpu.h"
#include "stlq/core/threading.h"
#include "stlq/io/base_store.h"
#include "stlq/io/bvecs_reader.h"
#include "stlq/io/fbin_reader.h"
#include "stlq/io/fvecs_reader.h"
#include "stlq/io/dataset_io.h"
#include "stlq/common/logger.h"
#include "stlq/pipeline/app_utils.h"
#include "stlq/quantizer/beam_search.h"
#include "stlq/quantizer/cost_utils.h"
#include "stlq/quantizer/icm.h"
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
#include "stlq/quantizer/icm_cuda.h"
#endif
#include "stlq/quantizer/least_squares.h"
#include "stlq/quantizer/precomp_large_root.h"
#include "stlq/quantizer/linear_algebra.h"
#include "stlq/common/timer.h"

namespace stlq
{
    namespace
    {
        // Print ICM backend line at most once per process (separately for F32 and U8 paths).
        // This avoids repeating identical "ICM backend" headers on every global-R iteration / baseset pass.
        struct BaseIcmBackendOnce
        {
            std::atomic<bool> did_f32{false};
            std::atomic<bool> did_u8{false};

            void MaybeLog(bool f32_prefix, const std::string& msg) {
                if (f32_prefix) {
                    bool expected = false;
                    if (!did_f32.compare_exchange_strong(expected, true, std::memory_order_relaxed)) return;
                }
                else {
                    bool expected = false;
                    if (!did_u8.compare_exchange_strong(expected, true, std::memory_order_relaxed)) return;
                }
                LogInfo(msg);
            }
        };

        BaseIcmBackendOnce g_base_icm_backend_once;

        template <typename T>
        struct PrefetchedBasicBlock
        {
            std::uint64_t start = 0;
            std::uint32_t count = 0;
            ColMajorMatrix<T> X; // d×count (column-major)
        };

        template <typename ReaderT, typename T>
        class AsyncBasicBlockPrefetcher
        {
        public:
            AsyncBasicBlockPrefetcher(const ReaderT& reader,
                                      std::uint64_t start0,
                                      std::uint64_t end,
                                      int d,
                                      std::uint64_t block_cols,
                                      int depth,
                                      int budget_mb)
                : reader_(&reader),
                  start0_(start0),
                  end_(end),
                  d_(d),
                  block_cols_(std::max<std::uint64_t>(1, block_cols)),
                  req_depth_(std::max(1, depth)),
                  budget_mb_(std::max(0, budget_mb)) {
            }

            ~AsyncBasicBlockPrefetcher() { Stop(); }

            AsyncBasicBlockPrefetcher(const AsyncBasicBlockPrefetcher&) = delete;
            AsyncBasicBlockPrefetcher& operator=(const AsyncBasicBlockPrefetcher&) = delete;

            int effective_depth() const { return eff_depth_; }

            void Start() {
                stop_.store(false);
                done_.store(false);
                {
                    std::lock_guard<std::mutex> guard(mu_);
                    err_msg_.clear();
                }
                eff_depth_ = ComputeEffectiveDepth();
                worker_ = std::thread([this]() { this->Run(); });
            }

            void Stop() {
                stop_.store(true);
                {
                    std::lock_guard<std::mutex> guard(mu_);
                    cv_not_empty_.notify_all();
                    cv_not_full_.notify_all();
                }
                if (worker_.joinable()) {
                    worker_.join();
                }
            }

            bool Pop(PrefetchedBasicBlock<T>* out) {
                if (!out) return false;
                std::unique_lock<std::mutex> lock(mu_);
                cv_not_empty_.wait(lock, [&]()
                {
                    return stop_.load() || !queue_.empty() || done_.load();
                });
                if (stop_.load()) return false;
                if (queue_.empty()) return false;
                *out = std::move(queue_.front());
                queue_.pop_front();
                cv_not_full_.notify_one();
                return true;
            }

            std::string error() const {
                std::lock_guard<std::mutex> guard(mu_);
                return err_msg_;
            }

        private:
            int ComputeEffectiveDepth() const {
                int depth = req_depth_;
                if (budget_mb_ <= 0) return depth;
                const std::uint64_t max_cols = block_cols_;
                const std::uint64_t bytes_per_col = static_cast<std::uint64_t>(d_) * sizeof(T);
                const std::uint64_t bytes_per_block = max_cols * bytes_per_col;
                if (bytes_per_block == 0) return 1;
                const std::uint64_t budget_b = static_cast<std::uint64_t>(budget_mb_) * 1024ull * 1024ull;
                const std::uint64_t max_depth64 = budget_b / bytes_per_block;
                const int max_depth = static_cast<int>(std::max<std::uint64_t>(
                    1, std::min<std::uint64_t>(max_depth64, 1ull << 30)));
                depth = std::max(1, std::min(depth, max_depth));
                return depth;
            }

            void Fail(const std::string& msg) {
                stop_.store(true);
                {
                    std::lock_guard<std::mutex> guard(mu_);
                    if (err_msg_.empty()) {
                        err_msg_ = msg;
                    }
                }
                cv_not_empty_.notify_all();
                cv_not_full_.notify_all();
            }

            void Run() {
                if (!reader_) {
                    Fail("AsyncBasicBlockPrefetcher: null reader.");
                    done_.store(true);
                    return;
                }
                std::uint64_t start = start0_;
                while (!stop_.load() && start < end_) {
                    const std::uint64_t remaining = end_ - start;
                    const auto count =
                        static_cast<std::uint32_t>(std::min<std::uint64_t>(remaining, block_cols_));
                    PrefetchedBasicBlock<T> item;
                    item.start = start;
                    item.count = count;
                    item.X.rows = d_;
                    item.X.cols = static_cast<int>(count);
                    item.X.data.resize(static_cast<std::size_t>(d_) * static_cast<std::size_t>(count));
                    std::string local_err;
                    const bool ok = reader_->ReadBlockInto(start, count,
                                                           item.X.data.data(),
                                                           item.X.data.size() * sizeof(T),
                                                           &local_err);
                    if (!ok) {
                        Fail(local_err.empty() ? "AsyncBasicBlockPrefetcher: ReadBlockInto failed." : local_err);
                        break;
                    }

                    std::unique_lock<std::mutex> lock(mu_);
                    cv_not_full_.wait(lock, [&]()
                    {
                        return stop_.load() || done_.load() ||
                            static_cast<int>(queue_.size()) < eff_depth_;
                    });
                    if (stop_.load()) break;
                    queue_.push_back(std::move(item));
                    cv_not_empty_.notify_one();
                    start += static_cast<std::uint64_t>(count);
                }
                done_.store(true);
                cv_not_empty_.notify_all();
            }

            const ReaderT* reader_ = nullptr;
            std::uint64_t start0_ = 0;
            std::uint64_t end_ = 0;
            int d_ = 0;
            std::uint64_t block_cols_ = 1;
            int req_depth_ = 2;
            int budget_mb_ = 0;
            int eff_depth_ = 2;

            std::atomic<bool> stop_{false};
            std::atomic<bool> done_{false};
            std::thread worker_;

            mutable std::mutex mu_;
            std::condition_variable cv_not_empty_;
            std::condition_variable cv_not_full_;
            std::deque<PrefetchedBasicBlock<T>> queue_;
            std::string err_msg_;
        };

        template <typename T>
        class BoundedQueue
        {
        public:
            explicit BoundedQueue(int capacity) : capacity_(std::max(1, capacity)) {
            }

            BoundedQueue(const BoundedQueue&) = delete;
            BoundedQueue& operator=(const BoundedQueue&) = delete;

            void Close() {
                {
                    std::lock_guard<std::mutex> guard(mu_);
                    closed_ = true;
                }
                cv_not_empty_.notify_all();
                cv_not_full_.notify_all();
            }

            bool Push(T&& item) {
                std::unique_lock<std::mutex> lock(mu_);
                cv_not_full_.wait(lock, [&]() { return closed_ || static_cast<int>(q_.size()) < capacity_; });
                if (closed_) return false;
                q_.push_back(std::move(item));
                cv_not_empty_.notify_one();
                return true;
            }

            bool Pop(T* out) {
                if (!out) return false;
                std::unique_lock<std::mutex> lock(mu_);
                cv_not_empty_.wait(lock, [&]() { return closed_ || !q_.empty(); });
                if (q_.empty()) return false;
                *out = std::move(q_.front());
                q_.pop_front();
                cv_not_full_.notify_one();
                return true;
            }

            bool TryPop(T* out) {
                if (!out) return false;
                std::lock_guard<std::mutex> guard(mu_);
                if (q_.empty()) return false;
                *out = std::move(q_.front());
                q_.pop_front();
                cv_not_full_.notify_one();
                return true;
            }

        private:
            int capacity_ = 1;
            bool closed_ = false;
            std::mutex mu_;
            std::condition_variable cv_not_empty_;
            std::condition_variable cv_not_full_;
            std::deque<T> q_;
        };

        inline int ClampDepthByBudget(int depth, int budget_mb, std::uint64_t bytes_per_block) {
            int eff = std::max(1, depth);
            if (budget_mb <= 0 || bytes_per_block == 0) return eff;
            const std::uint64_t budget_b = static_cast<std::uint64_t>(budget_mb) * 1024ull * 1024ull;
            const std::uint64_t max_depth64 = budget_b / bytes_per_block;
            const int max_depth = static_cast<int>(std::max<std::uint64_t>(
                1, std::min<std::uint64_t>(max_depth64, 1ull << 30)));
            eff = std::max(1, std::min(eff, max_depth));
            return eff;
        }

        float MeanMseFromCosts(const std::vector<float>& cost) {
            if (cost.empty()) {
                return 0.0f;
            }
            double s = 0.0;
#pragma omp parallel for default(none) schedule(static) reduction(+:s) shared(cost)
            for (std::int64_t i = 0; i < static_cast<std::int64_t>(cost.size()); ++i) {
                s += static_cast<double>(cost[static_cast<std::size_t>(i)]);
            }
            return static_cast<float>(s / static_cast<double>(cost.size()));
        }

        inline void ConvertU8TileToF32(const std::uint8_t* src,
                                       int d,
                                       int n,
                                       float* dst) {
            // src: column-major (ld=d), n columns.
            // dst: column-major (ld=d), n columns.
            for (int j = 0; j < n; ++j) {
                const std::uint8_t* s = src + static_cast<std::size_t>(j) * static_cast<std::size_t>(d);
                float* o = dst + static_cast<std::size_t>(j) * static_cast<std::size_t>(d);
#pragma omp simd
                for (int r = 0; r < d; ++r) {
                    o[r] = static_cast<float>(s[r]);
                }
            }
        }

        struct BaseTileBuf
        {
            // Fixed capacity at tile_n; per-tile we only change `.cols` and reuse storage.
            ColMajorMatrix<float> Xtmp; // d×tile_n (u8 path only; safe to allocate anyway)
            ColMajorMatrix<float> Xblk; // d×tile_n
            ColMajorMatrix<float> xC_full; // H_full×tile_n (full-precomp branch)
            ColMajorMatrix<float> xC_small; // H_small×tile_n (large-root branch)

            ColMajorMatrix<FullCode> B_full; // m×tile_n (full-precomp branch)
            ColMajorMatrix<Code> B_small; // (m-1)×tile_n
            ColMajorMatrix<float> a; // m×tile_n
            std::vector<std::uint32_t> cluster_id; // tile_n (resized to tlen before calling APIs requiring exact size)

            std::vector<float> X_norm2; // tile_n
            std::vector<float> cost_beam; // tile_n

            ColMajorMatrix<std::uint8_t> x_u8_tile; // d×tile_n (optional raw bucket)

            void Init(int d, int tile_n, int H_full, int H_small, int m, bool need_full_precomp, bool need_large_root) {
                Xtmp = ColMajorMatrix<float>(d, tile_n);
                Xblk = ColMajorMatrix<float>(d, tile_n);
                if (need_full_precomp) {
                    xC_full = ColMajorMatrix<float>(H_full, tile_n);
                    B_full = ColMajorMatrix<FullCode>(m, tile_n);
                }
                if (need_large_root) {
                    xC_small = ColMajorMatrix<float>(H_small, tile_n);
                }
                B_small = ColMajorMatrix<Code>(std::max(0, m - 1), tile_n);
                a = ColMajorMatrix<float>(m, tile_n);
                cluster_id.resize(static_cast<std::size_t>(tile_n));
                X_norm2.resize(static_cast<std::size_t>(tile_n));
                cost_beam.resize(static_cast<std::size_t>(tile_n));

                x_u8_tile.rows = d;
                x_u8_tile.cols = tile_n;
                x_u8_tile.data.resize(static_cast<std::size_t>(d) * static_cast<std::size_t>(tile_n));
            }

            void SetLen(int tlen, bool need_full_precomp, bool need_large_root) {
                Xtmp.cols = tlen;
                Xblk.cols = tlen;
                if (need_full_precomp) {
                    xC_full.cols = tlen;
                    B_full.cols = tlen;
                }
                if (need_large_root) {
                    xC_small.cols = tlen;
                }
                B_small.cols = tlen;
                a.cols = tlen;
                // cluster_id/cost/X_norm2 are std::vector: keep capacity; resize only when API requires exact size.
            }
        };

        struct BasicWriteTileJobU8
        {
            std::uint64_t global_start = 0;
            int tlen = 0;
            std::vector<std::uint32_t> cluster_id; // tlen
            ColMajorMatrix<Code> B_small; // (m-1)×tlen
            ColMajorMatrix<float> a; // m×tlen
            std::vector<std::uint8_t> raw_u8; // d×tlen col-major (optional)
        };

        struct BasicWriteTileJobF32
        {
            std::uint64_t global_start = 0;
            int tlen = 0;
            std::vector<std::uint32_t> cluster_id; // tlen
            ColMajorMatrix<Code> B_small; // (m-1)×tlen
            ColMajorMatrix<float> a; // m×tlen
            std::vector<float> raw_f32; // d×tlen col-major (optional)
        };

        class AsyncBasicTileWriterU8
        {
        public:
            AsyncBasicTileWriterU8(io::BaseBasicWriter* writer,
                                   int d,
                                   bool write_raw_bucket,
                                   int depth,
                                   int budget_mb,
                                   int m,
                                   int tile_n,
                                   std::string* err_out)
                : writer_(writer),
                  d_(d),
                  write_raw_bucket_(write_raw_bucket),
                  err_out_(err_out) {
                if (!writer_) return;
                std::uint64_t bytes_per_tile = 0;
                bytes_per_tile += static_cast<std::uint64_t>(tile_n) * sizeof(std::uint32_t);
                bytes_per_tile += static_cast<std::uint64_t>(std::max(0, m - 1)) * static_cast<std::uint64_t>(tile_n) *
                    sizeof(Code);
                bytes_per_tile += static_cast<std::uint64_t>(m) * static_cast<std::uint64_t>(tile_n) * sizeof(float);
                if (write_raw_bucket_) {
                    bytes_per_tile += static_cast<std::uint64_t>(d_) * static_cast<std::uint64_t>(tile_n) * sizeof(
                        std::uint8_t);
                }
                const int eff_depth = ClampDepthByBudget(depth, budget_mb, bytes_per_tile);
                q_ = std::make_unique<BoundedQueue<BasicWriteTileJobU8>>(eff_depth);
                worker_ = std::thread([&] { WorkerLoop(); });
            }

            ~AsyncBasicTileWriterU8() { Stop(); }

            AsyncBasicTileWriterU8(const AsyncBasicTileWriterU8&) = delete;
            AsyncBasicTileWriterU8& operator=(const AsyncBasicTileWriterU8&) = delete;

            [[nodiscard]] bool Enabled() const { return writer_ && worker_.joinable(); }

            bool Push(BasicWriteTileJobU8&& job) {
                if (!Enabled()) return false;
                if (failed_.load(std::memory_order_relaxed)) return false;
                if (!q_->Push(std::move(job))) return false;
                return !failed_.load(std::memory_order_relaxed);
            }

            bool Finish() {
                if (!Enabled()) return true;
                q_->Close();
                if (worker_.joinable()) worker_.join();
                if (failed_.load(std::memory_order_relaxed)) {
                    if (err_out_ && err_out_->empty()) {
                        *err_out_ = worker_err_.empty() ? "AsyncBasicTileWriterU8: writer failed." : worker_err_;
                    }
                    return false;
                }
                return true;
            }

            void Stop() {
                if (!Enabled()) return;
                q_->Close();
                if (worker_.joinable()) worker_.join();
            }

        private:
            void Fail(const std::string& msg) {
                failed_.store(true, std::memory_order_relaxed);
                worker_err_ = msg;
                if (q_) q_->Close();
            }

            void WorkerLoop() {
                BasicWriteTileJobU8 job;
                while (q_ && q_->Pop(&job)) {
                    std::string local_err;
                    if (write_raw_bucket_) {
                        if (!writer_->AppendBlockRawU8(job.global_start,
                                                       job.cluster_id,
                                                       job.B_small,
                                                       job.a,
                                                       job.raw_u8.data(),
                                                       /*ld_x_u8=*/d_,
                                                       &local_err)) {
                            Fail(local_err.empty() ? "AppendBlockRawU8 failed." : local_err);
                            break;
                        }
                    }
                    else {
                        if (!writer_->AppendBlock(job.global_start,
                                                  job.cluster_id,
                                                  job.B_small,
                                                  job.a,
                                                  /*x_u8_optional=*/nullptr,
                                                  &local_err)) {
                            Fail(local_err.empty() ? "AppendBlock failed." : local_err);
                            break;
                        }
                    }
                    job = BasicWriteTileJobU8{};
                }
            }

            io::BaseBasicWriter* writer_ = nullptr;
            int d_ = 0;
            bool write_raw_bucket_ = false;
            std::string* err_out_ = nullptr;

            std::unique_ptr<BoundedQueue<BasicWriteTileJobU8>> q_;
            std::thread worker_;
            std::atomic<bool> failed_{false};
            std::string worker_err_;
        };

        class AsyncBasicTileWriterF32
        {
        public:
            AsyncBasicTileWriterF32(io::BaseBasicWriter* writer,
                                    int d,
                                    bool write_raw_bucket,
                                    int depth,
                                    int budget_mb,
                                    int m,
                                    int tile_n,
                                    std::string* err_out)
                : writer_(writer),
                  d_(d),
                  write_raw_bucket_(write_raw_bucket),
                  err_out_(err_out) {
                if (!writer_) return;
                std::uint64_t bytes_per_tile = 0;
                bytes_per_tile += static_cast<std::uint64_t>(tile_n) * sizeof(std::uint32_t);
                bytes_per_tile += static_cast<std::uint64_t>(std::max(0, m - 1)) * static_cast<std::uint64_t>(tile_n) *
                    sizeof(Code);
                bytes_per_tile += static_cast<std::uint64_t>(m) * static_cast<std::uint64_t>(tile_n) * sizeof(float);
                if (write_raw_bucket_) {
                    bytes_per_tile += static_cast<std::uint64_t>(d_) * static_cast<std::uint64_t>(tile_n) * sizeof(
                        float);
                }
                const int eff_depth = ClampDepthByBudget(depth, budget_mb, bytes_per_tile);
                q_ = std::make_unique<BoundedQueue<BasicWriteTileJobF32>>(eff_depth);
                worker_ = std::thread([&] { WorkerLoop(); });
            }

            ~AsyncBasicTileWriterF32() { Stop(); }

            AsyncBasicTileWriterF32(const AsyncBasicTileWriterF32&) = delete;
            AsyncBasicTileWriterF32& operator=(const AsyncBasicTileWriterF32&) = delete;

            [[nodiscard]] bool Enabled() const { return writer_ && worker_.joinable(); }

            bool Push(BasicWriteTileJobF32&& job) {
                if (!Enabled()) return false;
                if (failed_.load(std::memory_order_relaxed)) return false;
                if (!q_->Push(std::move(job))) return false;
                return !failed_.load(std::memory_order_relaxed);
            }

            bool Finish() {
                if (!Enabled()) return true;
                q_->Close();
                if (worker_.joinable()) worker_.join();
                if (failed_.load(std::memory_order_relaxed)) {
                    if (err_out_ && err_out_->empty()) {
                        *err_out_ = worker_err_.empty() ? "AsyncBasicTileWriterF32: writer failed." : worker_err_;
                    }
                    return false;
                }
                return true;
            }

            void Stop() {
                if (!Enabled()) return;
                q_->Close();
                if (worker_.joinable()) worker_.join();
            }

        private:
            void Fail(const std::string& msg) {
                failed_.store(true, std::memory_order_relaxed);
                worker_err_ = msg;
                if (q_) q_->Close();
            }

            void WorkerLoop() {
                BasicWriteTileJobF32 job;
                while (q_ && q_->Pop(&job)) {
                    std::string local_err;
                    if (write_raw_bucket_) {
                        if (!writer_->AppendBlockRawF32(job.global_start,
                                                        job.cluster_id,
                                                        job.B_small,
                                                        job.a,
                                                        job.raw_f32.data(),
                                                        /*ld_x_f32=*/d_,
                                                        &local_err)) {
                            Fail(local_err.empty() ? "AppendBlockRawF32 failed." : local_err);
                            break;
                        }
                    }
                    else {
                        if (!writer_->AppendBlock(job.global_start,
                                                  job.cluster_id,
                                                  job.B_small,
                                                  job.a,
                                                  /*x_u8_optional=*/nullptr,
                                                  &local_err)) {
                            Fail(local_err.empty() ? "AppendBlock failed." : local_err);
                            break;
                        }
                    }
                    job = BasicWriteTileJobF32{};
                }
            }

            io::BaseBasicWriter* writer_ = nullptr;
            int d_ = 0;
            bool write_raw_bucket_ = false;
            std::string* err_out_ = nullptr;

            std::unique_ptr<BoundedQueue<BasicWriteTileJobF32>> q_;
            std::thread worker_;
            std::atomic<bool> failed_{false};
            std::string worker_err_;
        };

        void ExtractClusterIdAndCodesSmall(const ColMajorMatrix<FullCode>& B_full,
                                           std::vector<std::uint32_t>* cluster_id,
                                           ColMajorMatrix<Code>* B_small) {
            const int m = B_full.rows;
            const int n = B_full.cols;
            if (static_cast<int>(cluster_id->size()) < n) {
                cluster_id->resize(static_cast<std::size_t>(n));
            }
            B_small->rows = std::max(0, m - 1);
            B_small->cols = n;
            const std::size_t need =
                static_cast<std::size_t>(B_small->rows) * static_cast<std::size_t>(n);
            if (B_small->data.size() < need) {
                B_small->data.resize(need);
            }
            for (int i = 0; i < n; ++i) {
                (*cluster_id)[static_cast<std::size_t>(i)] = static_cast<std::uint32_t>(B_full(0, i));
                for (int l = 1; l < m; ++l) {
                    (*B_small)(l - 1, i) = B_full(l, i);
                }
            }
        }

        struct BasicHybridBlockTimings
        {
            double rotate = 0.0;
            double copy = 0.0;
            double xc = 0.0;
            double beam = 0.0;
            double beam_root_gemm = 0.0;
            double beam_root_update = 0.0;
            double beam_expand = 0.0;
            double ls = 0.0;
            double icm = 0.0;
        };

        struct BasicHybridBlockMetrics
        {
            double mse_sum_beam = 0.0;
            double mse_sum_final = 0.0;
            std::uint64_t n = 0;
        };

#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
        struct BasicHybridCudaIcmFlags
        {
            bool used_any = false;
            bool used_device_xc = false;
            bool used_device_x = false;
        };
#endif

        struct BasicHybridTaskU8
        {
            std::uint64_t block_id = 0;
            std::uint64_t start = 0;
            std::uint32_t nread = 0;
            ColMajorMatrix<std::uint8_t> x_u8; // d×nread
        };

        struct BasicHybridTaskF32
        {
            std::uint64_t block_id = 0;
            std::uint64_t start = 0;
            std::uint32_t nread = 0;
            ColMajorMatrix<float> x_f32; // d×nread
        };

        struct BasicHybridResultU8
        {
            std::uint64_t block_id = 0;
            std::uint64_t start = 0;
            std::uint32_t nread = 0;

            std::vector<std::uint32_t> cluster_id; // nread
            ColMajorMatrix<Code> B_small; // (m-1)×nread
            ColMajorMatrix<float> a; // m×nread

            // Optional: keep raw block for vector bucket records (d×nread).
            ColMajorMatrix<std::uint8_t> x_u8;

            BasicHybridBlockTimings t;
            BasicHybridBlockMetrics metrics;
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
            CudaIcmTiming icm_cuda{};
            BasicHybridCudaIcmFlags cuda_icm_flags{};
#endif
        };

        struct BasicHybridResultF32
        {
            std::uint64_t block_id = 0;
            std::uint64_t start = 0;
            std::uint32_t nread = 0;

            std::vector<std::uint32_t> cluster_id; // nread
            ColMajorMatrix<Code> B_small; // (m-1)×nread
            ColMajorMatrix<float> a; // m×nread

            // Optional: keep raw block for vector bucket records (d×nread).
            ColMajorMatrix<float> x_f32;

            BasicHybridBlockTimings t;
            BasicHybridBlockMetrics metrics;
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
            CudaIcmTiming icm_cuda{};
            BasicHybridCudaIcmFlags cuda_icm_flags{};
#endif
        };

        inline void CopyTileOutputsIntoBlock(int m,
                                             int t0,
                                             int tlen,
                                             const std::vector<std::uint32_t>& cluster_id_tile,
                                             const ColMajorMatrix<Code>& B_small_tile,
                                             const ColMajorMatrix<float>& a_tile,
                                             std::vector<std::uint32_t>* cluster_id_block,
                                             ColMajorMatrix<Code>* B_small_block,
                                             ColMajorMatrix<float>* a_block) {
            for (int j = 0; j < tlen; ++j) {
                const int dst_col = t0 + j;
                (*cluster_id_block)[static_cast<std::size_t>(dst_col)] = cluster_id_tile[static_cast<std::size_t>(j)];
                if (B_small_block && B_small_tile.rows > 0) {
                    std::memcpy(B_small_block->Col(dst_col),
                                B_small_tile.Col(j),
                                static_cast<std::size_t>(B_small_tile.rows) * sizeof(Code));
                }
                if (a_block) {
                    std::memcpy(a_block->Col(dst_col),
                                a_tile.Col(j),
                                static_cast<std::size_t>(m) * sizeof(float));
                }
            }
        }

        bool EncodeBasicHybridBlockU8(const Config& cfg,
                                      const ColMajorMatrix<float>& R,
                                      const Precomp& pre_full,
                                      const PrecompLargeRoot* pre_large,
                                      bool use_large_root,
                                      StreamKernelProvider* kernels,
                                      bool want_mse,
                                      BasicHybridTaskU8* task,
                                      BasicHybridResultU8* out,
                                      std::string* err) {
            if (!task || !out) return false;
            const int m = cfg.model.m;
            if (m <= 1) {
                if (err) *err = "EncodeBasicHybridBlockU8: model.m must be >= 2.";
                return false;
            }
            const int d = task->x_u8.rows;
            if (d <= 0 || task->nread == 0) {
                return true;
            }
            if (!kernels) {
                if (err) *err = "EncodeBasicHybridBlockU8: kernels is null.";
                return false;
            }

            BasicHybridResultU8 res;
            res.block_id = task->block_id;
            res.start = task->start;
            res.nread = task->nread;
            res.cluster_id.resize(static_cast<std::size_t>(task->nread));
            res.B_small = ColMajorMatrix<Code>(std::max(0, m - 1), static_cast<int>(task->nread));
            res.a = ColMajorMatrix<float>(m, static_cast<int>(task->nread));

            ColMajorMatrix<std::uint8_t> x_u8_local = std::move(task->x_u8);

            const std::uint64_t block = static_cast<std::uint64_t>(std::max(1, cfg.large.base_block));
            (void)block;
            const int tile_n = 8192;

            BaseTileBuf buf;
            buf.Init(d, tile_n, pre_full.H, use_large_root ? pre_large->H_small : 0, m,
                     /*need_full_precomp=*/!use_large_root,
                     /*need_large_root=*/use_large_root);

            Timer t;

            for (int t0 = 0; t0 < x_u8_local.cols; t0 += tile_n) {
                const int tlen = std::min(tile_n, x_u8_local.cols - t0);

                buf.SetLen(tlen, /*need_full_precomp=*/!use_large_root, /*need_large_root=*/use_large_root);
                ColMajorMatrix<float>& Xtmp = buf.Xtmp;
                ColMajorMatrix<float>& Xblk = buf.Xblk;
                t.Reset();
                const std::uint8_t* src_u8 =
                    x_u8_local.data.data() + static_cast<std::size_t>(t0) * static_cast<std::size_t>(d);
                if (!kernels->IsGpu()) {
                    ConvertU8TileToF32(src_u8, d, tlen, Xtmp.data.data());
                    {
                        ScopedBlasThreads blas_scope(OmpMaxThreads());
                        GemmRaw(false, false,
                                d, tlen, d,
                                1.0f,
                                R.data.data(), R.rows,
                                Xtmp.data.data(), Xtmp.rows,
                                0.0f,
                                Xblk.data.data(), Xblk.rows);
                    }
                }
                else {
                    kernels->ConvertU8ToF32AndRotatePtr(src_u8, /*ld_in=*/d, /*rows=*/d, /*cols=*/tlen, R, &Xblk);
                }
                res.t.rotate += t.ElapsedSeconds();

                std::vector<std::uint32_t>& cluster_id = buf.cluster_id;
                ColMajorMatrix<Code>& B_small = buf.B_small;
                ColMajorMatrix<float>& a = buf.a;

                float beam_mse_block = 0.0f;
                float final_mse_block = 0.0f;

                if (!use_large_root) {
                    ColMajorMatrix<float>& xC = buf.xC_full;
                    t.Reset();
                    kernels->Gemm(true, false, 1.0f, pre_full.C_all, Xblk, 0.0f, &xC);
                    res.t.xc += t.ElapsedSeconds();

                    ColMajorMatrix<FullCode>& B_full = buf.B_full;
                    const int H_beam = std::max(1, cfg.base.encode.H_beam);
                    t.Reset();
                    BeamSearchPrefixLS(Xblk, pre_full, xC, H_beam, &B_full);
                    res.t.beam += t.ElapsedSeconds();

                    t.Reset();
                    ExtractClusterIdAndCodesSmall(B_full, &cluster_id, &B_small);
                    res.t.beam += t.ElapsedSeconds();

                    t.Reset();
                    SolveLeastSquaresAll(pre_full, xC, B_full, &a);
                    res.t.ls += t.ElapsedSeconds();

                    std::vector<float>& X_norm2 = buf.X_norm2;
                    std::vector<float>& cost_full = buf.cost_beam;
                    std::vector<float>* cost_full_ptr = want_mse ? &cost_full : nullptr;
                    if (want_mse) {
                        ComputeXNorm2(Xblk, &X_norm2);
                        ComputeCosts(pre_full, xC, B_full, a, X_norm2, &cost_full);
                        beam_mse_block = MeanMseFromCosts(cost_full);
                    }
                    else {
                        ComputeXNorm2(Xblk, &X_norm2);
                    }
                    const std::uint64_t sample_id_offset = res.start + static_cast<std::uint64_t>(t0);
                    t.Reset();
                    [[maybe_unused]] const bool want_cuda_icm =
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                        (cfg.runtime.use_cuda && kernels && kernels->IsGpu());
#else
                false;
#endif
                    if (cfg.base.encode.use_abs) {
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                        if (want_cuda_icm) {
                            CudaIcmTiming icm_t;
                            DynamicIcmWithIlsAbsNoNormalCuda(Xblk, pre_full, xC,
                                                             cfg.base.encode.icm_iters,
                                                             cfg.base.encode.ils_iters,
                                                             cfg.base.encode.perturb_k,
                                                             static_cast<std::uint32_t>(cfg.base.encode.seed),
                                                             &B_full, &a, &X_norm2, cost_full_ptr,
                                                             /*print_progress=*/false,
                                                             sample_id_offset,
                                                             cfg.large.profile_timing ? &icm_t : nullptr);
                            if (cfg.large.profile_timing) {
                                res.icm_cuda.upload_fixed += icm_t.upload_fixed;
                                res.icm_cuda.upload_B += icm_t.upload_B;
                                res.icm_cuda.icm_init += icm_t.icm_init;
                                res.icm_cuda.ils_copy_perturb += icm_t.ils_copy_perturb;
                                res.icm_cuda.ils_icm += icm_t.ils_icm;
                                res.icm_cuda.ils_accept += icm_t.ils_accept;
                                res.icm_cuda.download += icm_t.download;
                            }
                            res.cuda_icm_flags.used_any = true;
                        }
                        else
#endif
                        {
                            DynamicIcmWithIlsAbsNoNormal(Xblk, pre_full, xC,
                                                         cfg.base.encode.icm_iters,
                                                         cfg.base.encode.ils_iters,
                                                         cfg.base.encode.perturb_k,
                                                         static_cast<std::uint32_t>(cfg.base.encode.seed),
                                                         &B_full, &a, &X_norm2, cost_full_ptr,
                                                         /*print_progress=*/false,
                                                         sample_id_offset);
                        }
                    }
                    else {
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                        if (want_cuda_icm) {
                            CudaIcmTiming icm_t;
                            DynamicIcmWithIlsNoAbsNoNormalCuda(Xblk, pre_full, xC,
                                                               cfg.base.encode.icm_iters,
                                                               cfg.base.encode.ils_iters,
                                                               cfg.base.encode.perturb_k,
                                                               static_cast<std::uint32_t>(cfg.base.encode.seed),
                                                               &B_full, &a, &X_norm2, cost_full_ptr,
                                                               /*print_progress=*/false,
                                                               sample_id_offset,
                                                               cfg.large.profile_timing ? &icm_t : nullptr);
                            if (cfg.large.profile_timing) {
                                res.icm_cuda.upload_fixed += icm_t.upload_fixed;
                                res.icm_cuda.upload_B += icm_t.upload_B;
                                res.icm_cuda.icm_init += icm_t.icm_init;
                                res.icm_cuda.ils_copy_perturb += icm_t.ils_copy_perturb;
                                res.icm_cuda.ils_icm += icm_t.ils_icm;
                                res.icm_cuda.ils_accept += icm_t.ils_accept;
                                res.icm_cuda.download += icm_t.download;
                            }
                            res.cuda_icm_flags.used_any = true;
                        }
                        else
#endif
                        {
                            DynamicIcmWithIlsNoAbsNoNormal(Xblk, pre_full, xC,
                                                           cfg.base.encode.icm_iters,
                                                           cfg.base.encode.ils_iters,
                                                           cfg.base.encode.perturb_k,
                                                           static_cast<std::uint32_t>(cfg.base.encode.seed),
                                                           &B_full, &a, &X_norm2, cost_full_ptr,
                                                           /*print_progress=*/false,
                                                           sample_id_offset);
                        }
                    }
                    res.t.icm += t.ElapsedSeconds();

                    if (want_mse && cost_full_ptr) {
                        final_mse_block = MeanMseFromCosts(*cost_full_ptr);
                    }

                    // Update cluster_id/B_small after ICM if codes changed.
                    ExtractClusterIdAndCodesSmall(B_full, &cluster_id, &B_small);
                }
                else {
                    // Large-root path: keep root code as cluster_id, store only small codes.
                    ColMajorMatrix<float>& xC_small = buf.xC_small;
                    t.Reset();
                    DeviceMatF32View xC_dev{};
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                    const bool want_cuda_beam_xc =
                    (cfg.runtime.use_cuda &&
                        kernels && kernels->IsGpu() &&
                        (cfg.base.encode.H_beam == 2 || cfg.base.encode.H_beam == 4));
                    if (want_cuda_beam_xc) {
                        kernels->GemmDevice(true, false, 1.0f, pre_large->C_small, Xblk, 0.0f, &xC_dev);
                    }
                    else
#endif
                    {
                        kernels->Gemm(true, false, 1.0f, pre_large->C_small, Xblk, 0.0f, &xC_small);
                    }
                    res.t.xc += t.ElapsedSeconds();

                    const int H_beam = std::max(1, cfg.base.encode.H_beam);
                    t.Reset();
                    BeamSearchLargeRootTiming beam_lr;
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                    const bool want_cuda_icm_lr =
                        (cfg.runtime.use_cuda && kernels && kernels->IsGpu());
#else
            const bool want_cuda_icm_lr = false;
#endif
                    const bool want_gpu_beam_mse = want_mse && want_cuda_icm_lr && xC_dev.ptr;

#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                    const bool allow_tf32_beam = EffectiveCudaAllowTf32(cfg.runtime);
                    if (cfg.runtime.use_cuda) {
                        if (H_beam == 2) {
                            if (xC_dev.ptr) {
                                DeviceMatF32View X_dev{};
                                const bool have_X_dev = kernels && kernels->TryGetCachedDeviceF32(Xblk, &X_dev);
                                if (want_mse && want_cuda_icm_lr) {
                                    if (have_X_dev) {
                                        BeamSearchPrefixLSLargeRootCudaH2DeviceXcDeviceXWithBestErr(
                                            X_dev, *pre_large, xC_dev,
                                            allow_tf32_beam,
                                            &cluster_id, &B_small,
                                            &buf.cost_beam,
                                            cfg.large.profile_timing ? &beam_lr : nullptr);
                                    }
                                    else {
                                        BeamSearchPrefixLSLargeRootCudaH2DeviceXcWithBestErr(
                                            Xblk, *pre_large, xC_dev,
                                            allow_tf32_beam,
                                            &cluster_id, &B_small,
                                            &buf.cost_beam,
                                            cfg.large.profile_timing ? &beam_lr : nullptr);
                                    }
                                }
                                else {
                                    if (have_X_dev) {
                                        BeamSearchPrefixLSLargeRootCudaH2DeviceXcDeviceX(
                                            X_dev, *pre_large, xC_dev,
                                            allow_tf32_beam,
                                            &cluster_id, &B_small,
                                            cfg.large.profile_timing ? &beam_lr : nullptr);
                                    }
                                    else {
                                        BeamSearchPrefixLSLargeRootCudaH2DeviceXc(Xblk, *pre_large, xC_dev,
                                            allow_tf32_beam,
                                            &cluster_id, &B_small,
                                            cfg.large.profile_timing ? &beam_lr : nullptr);
                                    }
                                }
                            }
                            else {
                                BeamSearchPrefixLSLargeRootCudaH2(Xblk, *pre_large, xC_small,
                                                                  allow_tf32_beam,
                                                                  &cluster_id, &B_small,
                                                                  cfg.large.profile_timing ? &beam_lr : nullptr);
                            }
                        }
                        else if (H_beam == 4) {
                            if (xC_dev.ptr) {
                                DeviceMatF32View X_dev{};
                                const bool have_X_dev = kernels && kernels->TryGetCachedDeviceF32(Xblk, &X_dev);
                                if (want_mse && want_cuda_icm_lr) {
                                    if (have_X_dev) {
                                        BeamSearchPrefixLSLargeRootCudaH4DeviceXcDeviceXWithBestErr(
                                            X_dev, *pre_large, xC_dev,
                                            allow_tf32_beam,
                                            &cluster_id, &B_small,
                                            &buf.cost_beam,
                                            cfg.large.profile_timing ? &beam_lr : nullptr);
                                    }
                                    else {
                                        BeamSearchPrefixLSLargeRootCudaH4DeviceXcWithBestErr(
                                            Xblk, *pre_large, xC_dev,
                                            allow_tf32_beam,
                                            &cluster_id, &B_small,
                                            &buf.cost_beam,
                                            cfg.large.profile_timing ? &beam_lr : nullptr);
                                    }
                                }
                                else {
                                    if (have_X_dev) {
                                        BeamSearchPrefixLSLargeRootCudaH4DeviceXcDeviceX(
                                            X_dev, *pre_large, xC_dev,
                                            allow_tf32_beam,
                                            &cluster_id, &B_small,
                                            cfg.large.profile_timing ? &beam_lr : nullptr);
                                    }
                                    else {
                                        BeamSearchPrefixLSLargeRootCudaH4DeviceXc(Xblk, *pre_large, xC_dev,
                                            allow_tf32_beam,
                                            &cluster_id, &B_small,
                                            cfg.large.profile_timing ? &beam_lr : nullptr);
                                    }
                                }
                            }
                            else {
                                BeamSearchPrefixLSLargeRootCudaH4(Xblk, *pre_large, xC_small,
                                                                  allow_tf32_beam,
                                                                  &cluster_id, &B_small,
                                                                  cfg.large.profile_timing ? &beam_lr : nullptr);
                            }
                        }
                        else {
                            BeamSearchPrefixLSLargeRootCuda(Xblk, *pre_large, xC_small, H_beam,
                                                            allow_tf32_beam,
                                                            &cluster_id, &B_small,
                                                            cfg.large.profile_timing ? &beam_lr : nullptr);
                        }
                    }
                    else
#endif
                    {
                        BeamSearchPrefixLSLargeRoot(Xblk, *pre_large, xC_small, H_beam, &cluster_id, &B_small,
                                                    cfg.large.profile_timing ? &beam_lr : nullptr);
                    }
                    res.t.beam += t.ElapsedSeconds();
                    if (cfg.large.profile_timing) {
                        res.t.beam_root_gemm += beam_lr.root_gemm;
                        res.t.beam_root_update += beam_lr.root_update;
                        res.t.beam_expand += beam_lr.expand;
                    }

#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                    if (xC_dev.ptr && (!want_cuda_icm_lr || (want_mse && !want_gpu_beam_mse))) {
                        t.Reset();
                        kernels->DownloadF32(xC_dev, &xC_small);
                        res.t.xc += t.ElapsedSeconds();
                    }
#endif

                    if (!want_cuda_icm_lr || (want_mse && !want_gpu_beam_mse)) {
                        t.Reset();
                        SolveLeastSquaresAllLargeRoot(*pre_large, Xblk, xC_small, cluster_id, B_small, &a);
                        res.t.ls += t.ElapsedSeconds();
                    }

                    std::vector<float>& X_norm2 = buf.X_norm2;
                    std::vector<float>& cost_beam = buf.cost_beam;
                    if (want_mse) {
                        if (want_gpu_beam_mse) {
                            // cost_beam is filled by the CUDA beam search (best beam error per sample).
                            beam_mse_block = MeanMseFromCosts(cost_beam);
                        }
                        else {
                            ComputeXNorm2(Xblk, &X_norm2);
                            ComputeCostsLargeRoot(*pre_large, Xblk, xC_small, cluster_id, B_small, a,
                                                  X_norm2, &cost_beam);
                            beam_mse_block = MeanMseFromCosts(cost_beam);
                        }
                    }

                    if (!want_mse || want_gpu_beam_mse) {
                        ComputeXNorm2(Xblk, &X_norm2);
                    }

                    t.Reset();
                    std::vector<float>* cost_full_ptr = want_mse ? &buf.cost_beam : nullptr;
                    const std::uint64_t sample_id_offset = res.start + static_cast<std::uint64_t>(t0);
                    if (cfg.base.encode.use_abs) {
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                        if (want_cuda_icm_lr) {
                            CudaIcmTiming icm_t;
                            if (xC_dev.ptr) {
                                DeviceMatF32View X_dev{};
                                if (kernels->TryGetCachedDeviceF32(Xblk, &X_dev)) {
                                    res.cuda_icm_flags.used_any = true;
                                    res.cuda_icm_flags.used_device_xc = true;
                                    res.cuda_icm_flags.used_device_x = true;
                                    DynamicIcmWithIlsAbsNoNormalLargeRootCudaDeviceXcDeviceX(X_dev, *pre_large, xC_dev,
                                        cfg.base.encode.icm_iters,
                                        cfg.base.encode.ils_iters,
                                        cfg.base.encode.perturb_k,
                                        static_cast<std::uint32_t>(cfg.base.encode.seed),
                                        cluster_id,
                                        &B_small, &a, &X_norm2, cost_full_ptr,
                                        /*print_progress=*/false,
                                        sample_id_offset,
                                        cfg.large.profile_timing ? &icm_t : nullptr);
                                }
                                else {
                                    res.cuda_icm_flags.used_any = true;
                                    res.cuda_icm_flags.used_device_xc = true;
                                    DynamicIcmWithIlsAbsNoNormalLargeRootCudaDeviceXc(Xblk, *pre_large, xC_dev,
                                        cfg.base.encode.icm_iters,
                                        cfg.base.encode.ils_iters,
                                        cfg.base.encode.perturb_k,
                                        static_cast<std::uint32_t>(cfg.base.encode.seed),
                                        cluster_id,
                                        &B_small, &a, &X_norm2, cost_full_ptr,
                                        /*print_progress=*/false,
                                        sample_id_offset,
                                        cfg.large.profile_timing ? &icm_t : nullptr);
                                }
                            }
                            else {
                                res.cuda_icm_flags.used_any = true;
                                DynamicIcmWithIlsAbsNoNormalLargeRootCuda(Xblk, *pre_large, xC_small,
                                                                          cfg.base.encode.icm_iters,
                                                                          cfg.base.encode.ils_iters,
                                                                          cfg.base.encode.perturb_k,
                                                                          static_cast<std::uint32_t>(cfg.base.encode.
                                                                              seed),
                                                                          cluster_id,
                                                                          &B_small, &a, &X_norm2, cost_full_ptr,
                                                                          /*print_progress=*/false,
                                                                          sample_id_offset,
                                                                          cfg.large.profile_timing ? &icm_t : nullptr);
                            }
                            if (cfg.large.profile_timing) {
                                res.icm_cuda.upload_fixed += icm_t.upload_fixed;
                                res.icm_cuda.upload_B += icm_t.upload_B;
                                res.icm_cuda.icm_init += icm_t.icm_init;
                                res.icm_cuda.ils_copy_perturb += icm_t.ils_copy_perturb;
                                res.icm_cuda.ils_icm += icm_t.ils_icm;
                                res.icm_cuda.ils_accept += icm_t.ils_accept;
                                res.icm_cuda.download += icm_t.download;
                            }
                        }
                        else
#endif
                        {
                            DynamicIcmWithIlsAbsNoNormalLargeRoot(Xblk, *pre_large, xC_small,
                                                                  cfg.base.encode.icm_iters,
                                                                  cfg.base.encode.ils_iters,
                                                                  cfg.base.encode.perturb_k,
                                                                  static_cast<std::uint32_t>(cfg.base.encode.seed),
                                                                  cluster_id,
                                                                  &B_small, &a, &X_norm2, cost_full_ptr,
                                                                  /*print_progress=*/false,
                                                                  sample_id_offset);
                        }
                    }
                    else {
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                        if (want_cuda_icm_lr) {
                            CudaIcmTiming icm_t;
                            if (xC_dev.ptr) {
                                DeviceMatF32View X_dev{};
                                if (kernels->TryGetCachedDeviceF32(Xblk, &X_dev)) {
                                    res.cuda_icm_flags.used_any = true;
                                    res.cuda_icm_flags.used_device_xc = true;
                                    res.cuda_icm_flags.used_device_x = true;
                                    DynamicIcmWithIlsNoAbsNoNormalLargeRootCudaDeviceXcDeviceX(
                                        X_dev, *pre_large, xC_dev,
                                        cfg.base.encode.icm_iters,
                                        cfg.base.encode.ils_iters,
                                        cfg.base.encode.perturb_k,
                                        static_cast<std::uint32_t>(cfg.base.encode.seed),
                                        cluster_id,
                                        &B_small, &a, &X_norm2, cost_full_ptr,
                                        /*print_progress=*/false,
                                        sample_id_offset,
                                        cfg.large.profile_timing ? &icm_t : nullptr);
                                }
                                else {
                                    res.cuda_icm_flags.used_any = true;
                                    res.cuda_icm_flags.used_device_xc = true;
                                    DynamicIcmWithIlsNoAbsNoNormalLargeRootCudaDeviceXc(Xblk, *pre_large, xC_dev,
                                        cfg.base.encode.icm_iters,
                                        cfg.base.encode.ils_iters,
                                        cfg.base.encode.perturb_k,
                                        static_cast<std::uint32_t>(cfg.base.encode.seed),
                                        cluster_id,
                                        &B_small, &a, &X_norm2, cost_full_ptr,
                                        /*print_progress=*/false,
                                        sample_id_offset,
                                        cfg.large.profile_timing ? &icm_t : nullptr);
                                }
                            }
                            else {
                                res.cuda_icm_flags.used_any = true;
                                DynamicIcmWithIlsNoAbsNoNormalLargeRootCuda(Xblk, *pre_large, xC_small,
                                                                            cfg.base.encode.icm_iters,
                                                                            cfg.base.encode.ils_iters,
                                                                            cfg.base.encode.perturb_k,
                                                                            static_cast<std::uint32_t>(cfg.base.encode.
                                                                                seed),
                                                                            cluster_id,
                                                                            &B_small, &a, &X_norm2, cost_full_ptr,
                                                                            /*print_progress=*/false,
                                                                            sample_id_offset,
                                                                            cfg.large.profile_timing
                                                                                ? &icm_t
                                                                                : nullptr);
                            }
                            if (cfg.large.profile_timing) {
                                res.icm_cuda.upload_fixed += icm_t.upload_fixed;
                                res.icm_cuda.upload_B += icm_t.upload_B;
                                res.icm_cuda.icm_init += icm_t.icm_init;
                                res.icm_cuda.ils_copy_perturb += icm_t.ils_copy_perturb;
                                res.icm_cuda.ils_icm += icm_t.ils_icm;
                                res.icm_cuda.ils_accept += icm_t.ils_accept;
                                res.icm_cuda.download += icm_t.download;
                            }
                        }
                        else
#endif
                        {
                            DynamicIcmWithIlsNoAbsNoNormalLargeRoot(Xblk, *pre_large, xC_small,
                                                                    cfg.base.encode.icm_iters,
                                                                    cfg.base.encode.ils_iters,
                                                                    cfg.base.encode.perturb_k,
                                                                    static_cast<std::uint32_t>(cfg.base.encode.seed),
                                                                    cluster_id,
                                                                    &B_small, &a, &X_norm2, cost_full_ptr,
                                                                    /*print_progress=*/false,
                                                                    sample_id_offset);
                        }
                    }
                    res.t.icm += t.ElapsedSeconds();
                    if (want_mse && cost_full_ptr) {
                        final_mse_block = MeanMseFromCosts(*cost_full_ptr);
                    }
                }

                if (want_mse) {
                    res.metrics.mse_sum_beam += static_cast<double>(beam_mse_block) * static_cast<double>(tlen);
                    res.metrics.mse_sum_final += static_cast<double>(final_mse_block) * static_cast<double>(tlen);
                    res.metrics.n += static_cast<std::uint64_t>(tlen);
                }

                CopyTileOutputsIntoBlock(m, t0, tlen, cluster_id, B_small, a,
                                         &res.cluster_id, &res.B_small, &res.a);
            }

            if (cfg.large.write_vector_bucket) {
                res.x_u8 = std::move(x_u8_local);
            }

            *out = std::move(res);
            return true;
        }

        bool EncodeBasicHybridBlockF32(const Config& cfg,
                                       const ColMajorMatrix<float>& R,
                                       const Precomp& pre_full,
                                       const PrecompLargeRoot* pre_large,
                                       bool use_large_root,
                                       StreamKernelProvider* kernels,
                                       bool want_mse,
                                       BasicHybridTaskF32* task,
                                       BasicHybridResultF32* out,
                                       std::string* err) {
            if (!task || !out) return false;
            const int m = cfg.model.m;
            if (m <= 1) {
                if (err) *err = "EncodeBasicHybridBlockF32: model.m must be >= 2.";
                return false;
            }
            const int d = task->x_f32.rows;
            if (d <= 0 || task->nread == 0) {
                return true;
            }
            if (!kernels) {
                if (err) *err = "EncodeBasicHybridBlockF32: kernels is null.";
                return false;
            }

            BasicHybridResultF32 res;
            res.block_id = task->block_id;
            res.start = task->start;
            res.nread = task->nread;
            res.cluster_id.resize(static_cast<std::size_t>(task->nread));
            res.B_small = ColMajorMatrix<Code>(std::max(0, m - 1), static_cast<int>(task->nread));
            res.a = ColMajorMatrix<float>(m, static_cast<int>(task->nread));

            ColMajorMatrix<float> x_f32_local = std::move(task->x_f32);

            const int tile_n = 8192;
            BaseTileBuf buf;
            buf.Init(d, tile_n, pre_full.H, use_large_root ? pre_large->H_small : 0, m,
                     /*need_full_precomp=*/!use_large_root,
                     /*need_large_root=*/use_large_root);

            Timer t;

            for (int t0 = 0; t0 < x_f32_local.cols; t0 += tile_n) {
                const int tlen = std::min(tile_n, x_f32_local.cols - t0);
                buf.SetLen(tlen, /*need_full_precomp=*/!use_large_root, /*need_large_root=*/use_large_root);
                ColMajorMatrix<float>& Xblk = buf.Xblk;
                t.Reset();
                const float* x_ptr =
                    x_f32_local.data.data() + static_cast<std::size_t>(t0) * static_cast<std::size_t>(d);
                if (!kernels->IsGpu()) {
                    ScopedBlasThreads blas_scope(OmpMaxThreads());
                    GemmRaw(false, false,
                            d, tlen, d,
                            1.0f,
                            R.data.data(), R.rows,
                            x_ptr, d,
                            0.0f,
                            Xblk.data.data(), Xblk.rows);
                }
                else {
                    ColMajorMatrix<float>& Xtmp = buf.Xtmp;
                    Xtmp.rows = d;
                    Xtmp.cols = tlen;
                    std::memcpy(Xtmp.data.data(),
                                x_ptr,
                                sizeof(float) * static_cast<std::size_t>(d) * static_cast<std::size_t>(tlen));
                    kernels->Gemm(false, false, 1.0f, R, Xtmp, 0.0f, &Xblk);
                }
                res.t.rotate += t.ElapsedSeconds();

                std::vector<std::uint32_t>& cluster_id = buf.cluster_id;
                ColMajorMatrix<Code>& B_small = buf.B_small;
                ColMajorMatrix<float>& a = buf.a;

                float beam_mse_block = 0.0f;
                float final_mse_block = 0.0f;

                if (!use_large_root) {
                    ColMajorMatrix<float>& xC = buf.xC_full;
                    [[maybe_unused]] DeviceMatF32View xC_dev{};
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                    [[maybe_unused]] const bool want_cuda_icm =
                        (cfg.runtime.use_cuda && kernels && kernels->IsGpu());
#else
            [[maybe_unused]] const bool want_cuda_icm = false;
#endif
                    t.Reset();
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                    if (want_cuda_icm) {
                        kernels->GemmDevice(true, false, 1.0f, pre_full.C_all, Xblk, 0.0f, &xC_dev);
                        kernels->DownloadF32(xC_dev, &xC);
                    }
                    else
#endif
                    {
                        kernels->Gemm(true, false, 1.0f, pre_full.C_all, Xblk, 0.0f, &xC);
                    }
                    res.t.xc += t.ElapsedSeconds();

                    ColMajorMatrix<FullCode>& B_full = buf.B_full;
                    const int H_beam = std::max(1, cfg.base.encode.H_beam);
                    t.Reset();
                    BeamSearchPrefixLS(Xblk, pre_full, xC, H_beam, &B_full);
                    res.t.beam += t.ElapsedSeconds();

                    ExtractClusterIdAndCodesSmall(B_full, &cluster_id, &B_small);

                    t.Reset();
                    SolveLeastSquaresAll(pre_full, xC, B_full, &a);
                    res.t.ls += t.ElapsedSeconds();

                    std::vector<float>& cost_beam = buf.cost_beam;
                    std::vector<float>& X_norm2 = buf.X_norm2;
                    if (want_mse) {
                        ComputeXNorm2(Xblk, &X_norm2);
                        ComputeCosts(pre_full, xC, B_full, a, X_norm2, &cost_beam);
                        beam_mse_block = MeanMseFromCosts(cost_beam);
                    }

                    if (!want_mse) {
                        ComputeXNorm2(Xblk, &X_norm2);
                    }
                    std::vector<float>& cost_full = buf.cost_beam;
                    std::vector<float>* cost_full_ptr = want_mse ? &cost_full : nullptr;
                    const std::uint64_t sample_id_offset = res.start + static_cast<std::uint64_t>(t0);
                    t.Reset();
                    if (cfg.base.encode.use_abs) {
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                        if (want_cuda_icm) {
                            CudaIcmTiming icm_t;
                            if (xC_dev.ptr) {
                                DynamicIcmWithIlsAbsNoNormalCudaDeviceXc(Xblk, pre_full, xC_dev,
                                                                         cfg.base.encode.icm_iters,
                                                                         cfg.base.encode.ils_iters,
                                                                         cfg.base.encode.perturb_k,
                                                                         static_cast<std::uint32_t>(cfg.base.encode.
                                                                             seed),
                                                                         &B_full, &a, &X_norm2, cost_full_ptr,
                                                                         /*print_progress=*/false,
                                                                         sample_id_offset,
                                                                         cfg.large.profile_timing ? &icm_t : nullptr);
                            }
                            else {
                                DynamicIcmWithIlsAbsNoNormalCuda(Xblk, pre_full, xC,
                                                                 cfg.base.encode.icm_iters,
                                                                 cfg.base.encode.ils_iters,
                                                                 cfg.base.encode.perturb_k,
                                                                 static_cast<std::uint32_t>(cfg.base.encode.seed),
                                                                 &B_full, &a, &X_norm2, cost_full_ptr,
                                                                 /*print_progress=*/false,
                                                                 sample_id_offset,
                                                                 cfg.large.profile_timing ? &icm_t : nullptr);
                            }
                            if (cfg.large.profile_timing) {
                                res.icm_cuda.upload_fixed += icm_t.upload_fixed;
                                res.icm_cuda.upload_B += icm_t.upload_B;
                                res.icm_cuda.icm_init += icm_t.icm_init;
                                res.icm_cuda.ils_copy_perturb += icm_t.ils_copy_perturb;
                                res.icm_cuda.ils_icm += icm_t.ils_icm;
                                res.icm_cuda.ils_accept += icm_t.ils_accept;
                                res.icm_cuda.download += icm_t.download;
                            }
                            res.cuda_icm_flags.used_any = true;
                            res.cuda_icm_flags.used_device_xc = (xC_dev.ptr != nullptr);
                        }
                        else
#endif
                        {
                            DynamicIcmWithIlsAbsNoNormal(Xblk, pre_full, xC,
                                                         cfg.base.encode.icm_iters,
                                                         cfg.base.encode.ils_iters,
                                                         cfg.base.encode.perturb_k,
                                                         static_cast<std::uint32_t>(cfg.base.encode.seed),
                                                         &B_full, &a, &X_norm2, cost_full_ptr,
                                                         /*print_progress=*/false,
                                                         sample_id_offset);
                        }
                    }
                    else {
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                        if (want_cuda_icm) {
                            CudaIcmTiming icm_t;
                            if (xC_dev.ptr) {
                                DynamicIcmWithIlsNoAbsNoNormalCudaDeviceXc(Xblk, pre_full, xC_dev,
                                                                           cfg.base.encode.icm_iters,
                                                                           cfg.base.encode.ils_iters,
                                                                           cfg.base.encode.perturb_k,
                                                                           static_cast<std::uint32_t>(cfg.base.encode.
                                                                               seed),
                                                                           &B_full, &a, &X_norm2, cost_full_ptr,
                                                                           /*print_progress=*/false,
                                                                           sample_id_offset,
                                                                           cfg.large.profile_timing ? &icm_t : nullptr);
                            }
                            else {
                                DynamicIcmWithIlsNoAbsNoNormalCuda(Xblk, pre_full, xC,
                                                                   cfg.base.encode.icm_iters,
                                                                   cfg.base.encode.ils_iters,
                                                                   cfg.base.encode.perturb_k,
                                                                   static_cast<std::uint32_t>(cfg.base.encode.seed),
                                                                   &B_full, &a, &X_norm2, cost_full_ptr,
                                                                   /*print_progress=*/false,
                                                                   sample_id_offset,
                                                                   cfg.large.profile_timing ? &icm_t : nullptr);
                            }
                            if (cfg.large.profile_timing) {
                                res.icm_cuda.upload_fixed += icm_t.upload_fixed;
                                res.icm_cuda.upload_B += icm_t.upload_B;
                                res.icm_cuda.icm_init += icm_t.icm_init;
                                res.icm_cuda.ils_copy_perturb += icm_t.ils_copy_perturb;
                                res.icm_cuda.ils_icm += icm_t.ils_icm;
                                res.icm_cuda.ils_accept += icm_t.ils_accept;
                                res.icm_cuda.download += icm_t.download;
                            }
                            res.cuda_icm_flags.used_any = true;
                            res.cuda_icm_flags.used_device_xc = (xC_dev.ptr != nullptr);
                        }
                        else
#endif
                        {
                            DynamicIcmWithIlsNoAbsNoNormal(Xblk, pre_full, xC,
                                                           cfg.base.encode.icm_iters,
                                                           cfg.base.encode.ils_iters,
                                                           cfg.base.encode.perturb_k,
                                                           static_cast<std::uint32_t>(cfg.base.encode.seed),
                                                           &B_full, &a, &X_norm2, cost_full_ptr,
                                                           /*print_progress=*/false,
                                                           sample_id_offset);
                        }
                    }
                    res.t.icm += t.ElapsedSeconds();
                    if (want_mse && cost_full_ptr) {
                        final_mse_block = MeanMseFromCosts(*cost_full_ptr);
                    }

                    // Update cluster_id/B_small after ICM if codes changed.
                    ExtractClusterIdAndCodesSmall(B_full, &cluster_id, &B_small);
                }
                else {
                    // Large-root path: keep root code as cluster_id, store only small codes.
                    ColMajorMatrix<float>& xC_small = buf.xC_small;
                    t.Reset();
                    DeviceMatF32View xC_dev{};
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                    const bool want_cuda_beam_xc =
                    (cfg.runtime.use_cuda &&
                        kernels && kernels->IsGpu() &&
                        (cfg.base.encode.H_beam == 2 || cfg.base.encode.H_beam == 4));
                    if (want_cuda_beam_xc) {
                        kernels->GemmDevice(true, false, 1.0f, pre_large->C_small, Xblk, 0.0f, &xC_dev);
                    }
                    else
#endif
                    {
                        kernels->Gemm(true, false, 1.0f, pre_large->C_small, Xblk, 0.0f, &xC_small);
                    }
                    res.t.xc += t.ElapsedSeconds();

                    const int H_beam = std::max(1, cfg.base.encode.H_beam);
                    t.Reset();
                    BeamSearchLargeRootTiming beam_lr;
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                    const bool want_cuda_icm_lr =
                        (cfg.runtime.use_cuda && kernels && kernels->IsGpu());
#else
            const bool want_cuda_icm_lr = false;
#endif
                    const bool want_gpu_beam_mse = want_mse && want_cuda_icm_lr && xC_dev.ptr;
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                    const bool allow_tf32_beam = EffectiveCudaAllowTf32(cfg.runtime);
                    if (cfg.runtime.use_cuda) {
                        if (H_beam == 2) {
                            if (xC_dev.ptr) {
                                DeviceMatF32View X_dev{};
                                const bool have_X_dev = kernels && kernels->TryGetCachedDeviceF32(Xblk, &X_dev);
                                if (want_gpu_beam_mse) {
                                    if (have_X_dev) {
                                        BeamSearchPrefixLSLargeRootCudaH2DeviceXcDeviceXWithBestErr(
                                            X_dev, *pre_large, xC_dev,
                                            allow_tf32_beam,
                                            &cluster_id, &B_small,
                                            &buf.cost_beam,
                                            cfg.large.profile_timing ? &beam_lr : nullptr);
                                    }
                                    else {
                                        BeamSearchPrefixLSLargeRootCudaH2DeviceXcWithBestErr(
                                            Xblk, *pre_large, xC_dev,
                                            allow_tf32_beam,
                                            &cluster_id, &B_small,
                                            &buf.cost_beam,
                                            cfg.large.profile_timing ? &beam_lr : nullptr);
                                    }
                                }
                                else {
                                    if (have_X_dev) {
                                        BeamSearchPrefixLSLargeRootCudaH2DeviceXcDeviceX(
                                            X_dev, *pre_large, xC_dev,
                                            allow_tf32_beam,
                                            &cluster_id, &B_small,
                                            cfg.large.profile_timing ? &beam_lr : nullptr);
                                    }
                                    else {
                                        BeamSearchPrefixLSLargeRootCudaH2DeviceXc(Xblk, *pre_large, xC_dev,
                                            allow_tf32_beam,
                                            &cluster_id, &B_small,
                                            cfg.large.profile_timing ? &beam_lr : nullptr);
                                    }
                                }
                            }
                            else {
                                BeamSearchPrefixLSLargeRootCudaH2(Xblk, *pre_large, xC_small,
                                                                  allow_tf32_beam,
                                                                  &cluster_id, &B_small,
                                                                  cfg.large.profile_timing ? &beam_lr : nullptr);
                            }
                        }
                        else if (H_beam == 4) {
                            if (xC_dev.ptr) {
                                DeviceMatF32View X_dev{};
                                const bool have_X_dev = kernels && kernels->TryGetCachedDeviceF32(Xblk, &X_dev);
                                if (want_gpu_beam_mse) {
                                    if (have_X_dev) {
                                        BeamSearchPrefixLSLargeRootCudaH4DeviceXcDeviceXWithBestErr(
                                            X_dev, *pre_large, xC_dev,
                                            allow_tf32_beam,
                                            &cluster_id, &B_small,
                                            &buf.cost_beam,
                                            cfg.large.profile_timing ? &beam_lr : nullptr);
                                    }
                                    else {
                                        BeamSearchPrefixLSLargeRootCudaH4DeviceXcWithBestErr(
                                            Xblk, *pre_large, xC_dev,
                                            allow_tf32_beam,
                                            &cluster_id, &B_small,
                                            &buf.cost_beam,
                                            cfg.large.profile_timing ? &beam_lr : nullptr);
                                    }
                                }
                                else {
                                    if (have_X_dev) {
                                        BeamSearchPrefixLSLargeRootCudaH4DeviceXcDeviceX(
                                            X_dev, *pre_large, xC_dev,
                                            allow_tf32_beam,
                                            &cluster_id, &B_small,
                                            cfg.large.profile_timing ? &beam_lr : nullptr);
                                    }
                                    else {
                                        BeamSearchPrefixLSLargeRootCudaH4DeviceXc(Xblk, *pre_large, xC_dev,
                                            allow_tf32_beam,
                                            &cluster_id, &B_small,
                                            cfg.large.profile_timing ? &beam_lr : nullptr);
                                    }
                                }
                            }
                            else {
                                BeamSearchPrefixLSLargeRootCudaH4(Xblk, *pre_large, xC_small,
                                                                  allow_tf32_beam,
                                                                  &cluster_id, &B_small,
                                                                  cfg.large.profile_timing ? &beam_lr : nullptr);
                            }
                        }
                        else {
                            BeamSearchPrefixLSLargeRootCuda(Xblk, *pre_large, xC_small, H_beam,
                                                            allow_tf32_beam,
                                                            &cluster_id, &B_small,
                                                            cfg.large.profile_timing ? &beam_lr : nullptr);
                        }
                    }
                    else
#endif
                    {
                        BeamSearchPrefixLSLargeRoot(Xblk, *pre_large, xC_small, H_beam, &cluster_id, &B_small,
                                                    cfg.large.profile_timing ? &beam_lr : nullptr);
                    }
                    res.t.beam += t.ElapsedSeconds();
                    if (cfg.large.profile_timing) {
                        res.t.beam_root_gemm += beam_lr.root_gemm;
                        res.t.beam_root_update += beam_lr.root_update;
                        res.t.beam_expand += beam_lr.expand;
                    }

#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                    if (xC_dev.ptr && (!want_cuda_icm_lr || (want_mse && !want_gpu_beam_mse))) {
                        t.Reset();
                        kernels->DownloadF32(xC_dev, &xC_small);
                        res.t.xc += t.ElapsedSeconds();
                    }
#endif

                    if (!want_cuda_icm_lr || (want_mse && !want_gpu_beam_mse)) {
                        t.Reset();
                        SolveLeastSquaresAllLargeRoot(*pre_large, Xblk, xC_small, cluster_id, B_small, &a);
                        res.t.ls += t.ElapsedSeconds();
                    }

                    std::vector<float>& X_norm2 = buf.X_norm2;
                    std::vector<float>& cost_beam = buf.cost_beam;
                    if (want_mse) {
                        if (want_gpu_beam_mse) {
                            beam_mse_block = MeanMseFromCosts(cost_beam);
                        }
                        else {
                            ComputeXNorm2(Xblk, &X_norm2);
                            ComputeCostsLargeRoot(*pre_large, Xblk, xC_small, cluster_id, B_small, a,
                                                  X_norm2, &cost_beam);
                            beam_mse_block = MeanMseFromCosts(cost_beam);
                        }
                    }

                    if (!want_mse || want_gpu_beam_mse) {
                        ComputeXNorm2(Xblk, &X_norm2);
                    }

                    t.Reset();
                    std::vector<float>* cost_full_ptr = want_mse ? &buf.cost_beam : nullptr;
                    const std::uint64_t sample_id_offset = res.start + static_cast<std::uint64_t>(t0);
                    if (cfg.base.encode.use_abs) {
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                        if (want_cuda_icm_lr) {
                            CudaIcmTiming icm_t;
                            if (xC_dev.ptr) {
                                DeviceMatF32View X_dev{};
                                if (kernels->TryGetCachedDeviceF32(Xblk, &X_dev)) {
                                    res.cuda_icm_flags.used_any = true;
                                    res.cuda_icm_flags.used_device_xc = true;
                                    res.cuda_icm_flags.used_device_x = true;
                                    DynamicIcmWithIlsAbsNoNormalLargeRootCudaDeviceXcDeviceX(X_dev, *pre_large, xC_dev,
                                        cfg.base.encode.icm_iters,
                                        cfg.base.encode.ils_iters,
                                        cfg.base.encode.perturb_k,
                                        static_cast<std::uint32_t>(cfg.base.encode.seed),
                                        cluster_id,
                                        &B_small, &a, &X_norm2, cost_full_ptr,
                                        /*print_progress=*/false,
                                        sample_id_offset,
                                        cfg.large.profile_timing ? &icm_t : nullptr);
                                }
                                else {
                                    res.cuda_icm_flags.used_any = true;
                                    res.cuda_icm_flags.used_device_xc = true;
                                    DynamicIcmWithIlsAbsNoNormalLargeRootCudaDeviceXc(Xblk, *pre_large, xC_dev,
                                        cfg.base.encode.icm_iters,
                                        cfg.base.encode.ils_iters,
                                        cfg.base.encode.perturb_k,
                                        static_cast<std::uint32_t>(cfg.base.encode.seed),
                                        cluster_id,
                                        &B_small, &a, &X_norm2, cost_full_ptr,
                                        /*print_progress=*/false,
                                        sample_id_offset,
                                        cfg.large.profile_timing ? &icm_t : nullptr);
                                }
                            }
                            else {
                                res.cuda_icm_flags.used_any = true;
                                DynamicIcmWithIlsAbsNoNormalLargeRootCuda(Xblk, *pre_large, xC_small,
                                                                          cfg.base.encode.icm_iters,
                                                                          cfg.base.encode.ils_iters,
                                                                          cfg.base.encode.perturb_k,
                                                                          static_cast<std::uint32_t>(cfg.base.encode.
                                                                              seed),
                                                                          cluster_id,
                                                                          &B_small, &a, &X_norm2, cost_full_ptr,
                                                                          /*print_progress=*/false,
                                                                          sample_id_offset,
                                                                          cfg.large.profile_timing ? &icm_t : nullptr);
                            }
                            if (cfg.large.profile_timing) {
                                res.icm_cuda.upload_fixed += icm_t.upload_fixed;
                                res.icm_cuda.upload_B += icm_t.upload_B;
                                res.icm_cuda.icm_init += icm_t.icm_init;
                                res.icm_cuda.ils_copy_perturb += icm_t.ils_copy_perturb;
                                res.icm_cuda.ils_icm += icm_t.ils_icm;
                                res.icm_cuda.ils_accept += icm_t.ils_accept;
                                res.icm_cuda.download += icm_t.download;
                            }
                        }
                        else
#endif
                        {
                            DynamicIcmWithIlsAbsNoNormalLargeRoot(Xblk, *pre_large, xC_small,
                                                                  cfg.base.encode.icm_iters,
                                                                  cfg.base.encode.ils_iters,
                                                                  cfg.base.encode.perturb_k,
                                                                  static_cast<std::uint32_t>(cfg.base.encode.seed),
                                                                  cluster_id,
                                                                  &B_small, &a, &X_norm2, cost_full_ptr,
                                                                  /*print_progress=*/false,
                                                                  sample_id_offset);
                        }
                    }
                    else {
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                        if (want_cuda_icm_lr) {
                            CudaIcmTiming icm_t;
                            if (xC_dev.ptr) {
                                DeviceMatF32View X_dev{};
                                if (kernels->TryGetCachedDeviceF32(Xblk, &X_dev)) {
                                    res.cuda_icm_flags.used_any = true;
                                    res.cuda_icm_flags.used_device_xc = true;
                                    res.cuda_icm_flags.used_device_x = true;
                                    DynamicIcmWithIlsNoAbsNoNormalLargeRootCudaDeviceXcDeviceX(
                                        X_dev, *pre_large, xC_dev,
                                        cfg.base.encode.icm_iters,
                                        cfg.base.encode.ils_iters,
                                        cfg.base.encode.perturb_k,
                                        static_cast<std::uint32_t>(cfg.base.encode.seed),
                                        cluster_id,
                                        &B_small, &a, &X_norm2, cost_full_ptr,
                                        /*print_progress=*/false,
                                        sample_id_offset,
                                        cfg.large.profile_timing ? &icm_t : nullptr);
                                }
                                else {
                                    res.cuda_icm_flags.used_any = true;
                                    res.cuda_icm_flags.used_device_xc = true;
                                    DynamicIcmWithIlsNoAbsNoNormalLargeRootCudaDeviceXc(Xblk, *pre_large, xC_dev,
                                        cfg.base.encode.icm_iters,
                                        cfg.base.encode.ils_iters,
                                        cfg.base.encode.perturb_k,
                                        static_cast<std::uint32_t>(cfg.base.encode.seed),
                                        cluster_id,
                                        &B_small, &a, &X_norm2, cost_full_ptr,
                                        /*print_progress=*/false,
                                        sample_id_offset,
                                        cfg.large.profile_timing ? &icm_t : nullptr);
                                }
                            }
                            else {
                                res.cuda_icm_flags.used_any = true;
                                DynamicIcmWithIlsNoAbsNoNormalLargeRootCuda(Xblk, *pre_large, xC_small,
                                                                            cfg.base.encode.icm_iters,
                                                                            cfg.base.encode.ils_iters,
                                                                            cfg.base.encode.perturb_k,
                                                                            static_cast<std::uint32_t>(cfg.base.encode.
                                                                                seed),
                                                                            cluster_id,
                                                                            &B_small, &a, &X_norm2, cost_full_ptr,
                                                                            /*print_progress=*/false,
                                                                            sample_id_offset,
                                                                            cfg.large.profile_timing
                                                                                ? &icm_t
                                                                                : nullptr);
                            }
                            if (cfg.large.profile_timing) {
                                res.icm_cuda.upload_fixed += icm_t.upload_fixed;
                                res.icm_cuda.upload_B += icm_t.upload_B;
                                res.icm_cuda.icm_init += icm_t.icm_init;
                                res.icm_cuda.ils_copy_perturb += icm_t.ils_copy_perturb;
                                res.icm_cuda.ils_icm += icm_t.ils_icm;
                                res.icm_cuda.ils_accept += icm_t.ils_accept;
                                res.icm_cuda.download += icm_t.download;
                            }
                        }
                        else
#endif
                        {
                            DynamicIcmWithIlsNoAbsNoNormalLargeRoot(Xblk, *pre_large, xC_small,
                                                                    cfg.base.encode.icm_iters,
                                                                    cfg.base.encode.ils_iters,
                                                                    cfg.base.encode.perturb_k,
                                                                    static_cast<std::uint32_t>(cfg.base.encode.seed),
                                                                    cluster_id,
                                                                    &B_small, &a, &X_norm2, cost_full_ptr,
                                                                    /*print_progress=*/false,
                                                                    sample_id_offset);
                        }
                    }
                    res.t.icm += t.ElapsedSeconds();
                    if (want_mse && cost_full_ptr) {
                        final_mse_block = MeanMseFromCosts(*cost_full_ptr);
                    }
                }

                if (want_mse) {
                    res.metrics.mse_sum_beam += static_cast<double>(beam_mse_block) * static_cast<double>(tlen);
                    res.metrics.mse_sum_final += static_cast<double>(final_mse_block) * static_cast<double>(tlen);
                    res.metrics.n += static_cast<std::uint64_t>(tlen);
                }

                CopyTileOutputsIntoBlock(m, t0, tlen, cluster_id, B_small, a,
                                         &res.cluster_id, &res.B_small, &res.a);
            }

            if (cfg.large.write_vector_bucket) {
                res.x_f32 = std::move(x_f32_local);
            }

            *out = std::move(res);
            return true;
        }
    } // namespace

    bool EncodeBaseStreaming(const Config& cfg,
                             const io::BvecsReader& base_reader,
                             const ColMajorMatrix<float>& R,
                             const CodebookPack& C_root,
                             const Precomp& pre_full,
                             const PrecompLargeRoot* pre_large,
                             StreamKernelProvider* kernels,
                             io::BaseBasicWriter* writer,
                             float* out_beam_mse,
                             float* out_final_mse,
                             std::string* err,
                             std::uint64_t start_id,
                             const StreamingBasicCommitHook* commit_hook) {
        if (!writer) {
            if (err) *err = "EncodeBaseStreaming: writer is null.";
            return false;
        }
        std::uint64_t nbase = base_reader.n();
        if (cfg.dataset.nbase_set && cfg.dataset.nbase > 0) {
            nbase = std::min<std::uint64_t>(nbase, static_cast<std::uint64_t>(cfg.dataset.nbase));
        }
        if (start_id > nbase) {
            if (err) *err = "EncodeBaseStreaming: start_id out of range.";
            return false;
        }
        const int d = base_reader.d();
        if (d <= 0 || nbase == 0) {
            if (err) *err = "EncodeBaseStreaming: empty base reader.";
            return false;
        }
        if (R.rows != d || R.cols != d) {
            if (err) *err = "EncodeBaseStreaming: rotation matrix size mismatch.";
            return false;
        }
        const int m = cfg.model.m;
        if (m <= 1) {
            if (err) *err = "EncodeBaseStreaming: model.m must be >= 2.";
            return false;
        }

        const bool use_large_root = (pre_large && pre_large->ready && pre_large->h_vec.size() == C_root.books.size());

        CpuStreamKernels cpu_kernels;
        if (!kernels) {
            kernels = &cpu_kernels;
        }

        const bool can_use_gpu_lane =
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
            (cfg.runtime.use_cuda && kernels && kernels->IsGpu());
#else
        false;
#endif
        const bool use_hybrid =
        (cfg.runtime.basic_hybrid_enable &&
            cfg.runtime.basic_hybrid_cpu_stride > 0 &&
            can_use_gpu_lane);

        // Avoid re-printing the same header logs for each global R-iteration.
        {
            static std::uint64_t last_tag = 0;
            static bool printed = false;
            std::uint64_t tag = 1469598103934665603ULL;
            auto Mix = [&](std::uint64_t v)
            {
                tag ^= v;
                tag *= 1099511628211ULL;
            };
            Mix(nbase);
            Mix(static_cast<std::uint64_t>(d));
            Mix(static_cast<std::uint64_t>(m));
            Mix(static_cast<std::uint64_t>(std::max(1, cfg.base.encode.H_beam)));
            Mix(static_cast<std::uint64_t>(std::max(0, cfg.base.encode.icm_iters)));
            Mix(static_cast<std::uint64_t>(std::max(0, cfg.base.encode.ils_iters)));
            Mix(static_cast<std::uint64_t>(std::max(0, cfg.base.encode.perturb_k)));
            Mix(static_cast<std::uint64_t>(cfg.base.encode.use_abs ? 1 : 0));
            Mix(static_cast<std::uint64_t>(cfg.base.encode.seed));
            Mix(static_cast<std::uint64_t>(std::max(1, cfg.large.base_block)));
            Mix(static_cast<std::uint64_t>(use_large_root ? 1 : 0));
            Mix(static_cast<std::uint64_t>(cfg.large.profile_timing ? 1 : 0));
            Mix(static_cast<std::uint64_t>(use_hybrid ? 1 : 0));
            if (use_hybrid) {
                Mix(static_cast<std::uint64_t>(std::max(0, cfg.runtime.basic_hybrid_cpu_stride)));
                Mix(static_cast<std::uint64_t>(std::max(0, cfg.runtime.basic_hybrid_cpu_threads)));
                Mix(static_cast<std::uint64_t>(std::max(0, cfg.runtime.basic_hybrid_reorder_depth)));
                Mix(static_cast<std::uint64_t>(std::max(0, cfg.runtime.basic_hybrid_inflight_mb)));
            }
            if (!printed || tag != last_tag) {
                printed = true;
                last_tag = tag;
                LogInfo("Streaming basic encode params: n=" + std::to_string(nbase) +
                    " d=" + std::to_string(d) +
                    " m=" + std::to_string(m) +
                    " H_beam=" + std::to_string(std::max(1, cfg.base.encode.H_beam)) +
                    " icm_iters=" + std::to_string(std::max(0, cfg.base.encode.icm_iters)) +
                    " ils_iters=" + std::to_string(std::max(0, cfg.base.encode.ils_iters)) +
                    " perturb_k=" + std::to_string(std::max(0, cfg.base.encode.perturb_k)) +
                    " use_abs=" + std::string(cfg.base.encode.use_abs ? "true" : "false") +
                    " seed=" + std::to_string(cfg.base.encode.seed));
                LogInfo("Streaming IO: block=" + std::to_string(std::max(1, cfg.large.base_block)) +
                    " tile_n=8192" +
                    " write_vector_bucket=" + std::string(cfg.large.write_vector_bucket ? "true" : "false") +
                    " write_basic_to_bucket=" + std::string(cfg.large.write_basic_to_bucket ? "true" : "false") +
                    " profile_timing=" + std::string(cfg.large.profile_timing ? "true" : "false"));
                if (use_large_root) {
                    LogInfo("Streaming path: large-root (h0=" + std::to_string(pre_large->h_vec[0]) +
                        " H_small=" + std::to_string(pre_large->H_small) + ")");
                }
                else {
                    LogInfo("Streaming path: full-precomp (H=" + std::to_string(pre_full.H) + ")");
                }
                if (use_hybrid) {
                    const int cpu_threads =
                        (cfg.runtime.basic_hybrid_cpu_threads > 0)
                            ? cfg.runtime.basic_hybrid_cpu_threads
                            : OmpMaxThreads();
                    LogInfo("Streaming basic hybrid: enabled (cpu_stride=" + std::to_string(
                            cfg.runtime.basic_hybrid_cpu_stride) +
                        " cpu_threads=" + std::to_string(cpu_threads) +
                        " reorder_depth=" + std::to_string(std::max(1, cfg.runtime.basic_hybrid_reorder_depth)) +
                        " inflight_mb=" + std::to_string(std::max(0, cfg.runtime.basic_hybrid_inflight_mb)) + ")");
                }
                else if (cfg.runtime.basic_hybrid_enable && cfg.runtime.basic_hybrid_cpu_stride > 0) {
                    LogInfo(
                        "Streaming basic hybrid: requested but disabled (requires runtime.use_cuda=true and CUDA stream kernels).");
                }
            }
        }

        // MSE reporting is optional; disable it to reduce overhead when metrics are not needed.
        const bool want_mse = cfg.train.log_metrics || (out_beam_mse != nullptr) || (out_final_mse != nullptr);

        const std::uint64_t block = static_cast<std::uint64_t>(std::max(1, cfg.large.base_block));
        const bool want_commit_hook = (commit_hook && commit_hook->OnCommit);
        if (want_commit_hook && block > 0 && (start_id % block) != 0) {
            if (err)
                *err =
                    "EncodeBaseStreaming: start_id must be a multiple of large.base_block when using checkpointing.";
            return false;
        }

        auto CommitBlock = [&](std::uint64_t next_id) -> bool
        {
            if (!want_commit_hook) return true;
            if (!writer->Flush(err)) return false;
            return commit_hook->OnCommit(next_id, writer, commit_hook->ctx, err);
        };
        if (use_hybrid) {
            const int m_codes = std::max(0, m - 1);
            std::uint64_t bytes_per_vec = sizeof(std::uint32_t) +
                static_cast<std::uint64_t>(m_codes) * sizeof(Code) +
                static_cast<std::uint64_t>(m) * sizeof(float);
            if (cfg.large.write_vector_bucket) {
                bytes_per_vec += static_cast<std::uint64_t>(d); // raw u8
            }
            const std::uint64_t bytes_per_block = bytes_per_vec * block;
            int depth = std::max(2, cfg.runtime.basic_hybrid_reorder_depth);
            depth = ClampDepthByBudget(depth, cfg.runtime.basic_hybrid_inflight_mb, bytes_per_block);
            if (depth < 2) {
                depth = 2;
            }

            const int cpu_stride = std::max(1, cfg.runtime.basic_hybrid_cpu_stride);
            const int cpu_phase = cpu_stride - 1; // keep early blocks on GPU to reduce ordered-writer stalls
            const int cpu_threads =
                (cfg.runtime.basic_hybrid_cpu_threads > 0) ? cfg.runtime.basic_hybrid_cpu_threads : OmpMaxThreads();

            CpuStreamKernels cpu_lane_kernels;
            BoundedQueue<BasicHybridTaskU8> q_gpu(depth);
            BoundedQueue<BasicHybridTaskU8> q_cpu(depth);
            BoundedQueue<BasicHybridResultU8> q_res(depth);

            std::atomic<bool> failed{false};
            std::mutex err_mu;
            std::string worker_err;
            auto Fail = [&](const std::string& msg)
            {
                failed.store(true);
                {
                    std::lock_guard<std::mutex> guard(err_mu);
                    if (worker_err.empty()) worker_err = msg;
                }
                q_gpu.Close();
                q_cpu.Close();
                q_res.Close();
            };

            std::thread gpu_worker([&]()
            {
                ScopedOmpThreads omp_scope(1);
                BasicHybridTaskU8 task;
                while (!failed.load() && q_gpu.Pop(&task)) {
                    BasicHybridResultU8 res;
                    std::string local_err;
                    if (!EncodeBasicHybridBlockU8(cfg, R, pre_full, pre_large, use_large_root, kernels, want_mse,
                                                  &task, &res, &local_err)) {
                        Fail(local_err.empty() ? "EncodeBaseStreaming(hybrid,u8): GPU lane failed." : local_err);
                        break;
                    }
                    if (!q_res.Push(std::move(res))) break;
                }
            });

            std::thread cpu_worker([&]()
            {
                ScopedOmpThreads omp_scope(cpu_threads);
                BasicHybridTaskU8 task;
                while (!failed.load() && q_cpu.Pop(&task)) {
                    BasicHybridResultU8 res;
                    std::string local_err;
                    if (!EncodeBasicHybridBlockU8(cfg, R, pre_full, pre_large, use_large_root, &cpu_lane_kernels,
                                                  want_mse,
                                                  &task, &res, &local_err)) {
                        Fail(local_err.empty() ? "EncodeBaseStreaming(hybrid,u8): CPU lane failed." : local_err);
                        break;
                    }
                    if (!q_res.Push(std::move(res))) break;
                }
            });

            std::vector<std::optional<BasicHybridResultU8>> ring(static_cast<std::size_t>(depth));
            const std::uint64_t first_block = (block > 0) ? (start_id / block) : 0;
            std::uint64_t next_block = first_block;
            std::uint64_t blocks_written = first_block;
            std::uint64_t inflight = 0;

            double t_read_total = 0.0;
            double t_write_total = 0.0;
            double t_rotate_total = 0.0;
            double t_copy_total = 0.0;
            double t_xc_total = 0.0;
            double t_beam_total = 0.0;
            double t_beam_root_gemm_total = 0.0;
            double t_beam_root_update_total = 0.0;
            double t_beam_expand_total = 0.0;
            double t_ls_total = 0.0;
            double t_icm_total = 0.0;
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
            CudaIcmTiming icm_cuda_total;
            bool used_cuda_icm_any = false;
            bool used_cuda_icm_any_device_xc = false;
            bool used_cuda_icm_any_device_x = false;
#else
        bool used_cuda_icm_any = false;
        bool used_cuda_icm_any_device_xc = false;
        bool used_cuda_icm_any_device_x = false;
#endif

            double mse_sum_beam = 0.0;
            double mse_sum_final = 0.0;
            std::uint64_t n_total = 0;

            Timer t_wall;
            const std::uint64_t blocks_total = (nbase + block - 1) / block;

            std::unique_ptr<AsyncBasicBlockPrefetcher<io::BvecsReader, std::uint8_t>> async_io;
            if (cfg.runtime.basic_async_io && cfg.runtime.basic_async_io_depth > 0) {
                async_io = std::make_unique<AsyncBasicBlockPrefetcher<io::BvecsReader, std::uint8_t>>(
                    base_reader, start_id, nbase, d, block,
                    cfg.runtime.basic_async_io_depth,
                    cfg.runtime.basic_async_io_mb);
                async_io->Start();
            }

            auto DrainOneResultBlocking = [&]() -> bool
            {
                BasicHybridResultU8 got;
                if (!q_res.Pop(&got)) return false;
                const std::uint64_t bid = got.block_id;
                const std::uint64_t max_bid = next_block + static_cast<std::uint64_t>(depth) - 1;
                if (bid < next_block || bid > max_bid) {
                    Fail("EncodeBaseStreaming(hybrid,u8): reorder depth too small (block_id out of window).");
                    return false;
                }
                const auto slot = static_cast<std::size_t>(bid % static_cast<std::uint64_t>(depth));
                if (ring[slot].has_value()) {
                    Fail("EncodeBaseStreaming(hybrid,u8): reorder ring slot collision (increase reorder depth).");
                    return false;
                }
                ring[slot] = std::move(got);

                while (next_block < blocks_total) {
                    const auto idx = static_cast<std::size_t>(next_block % static_cast<std::uint64_t>(depth));
                    if (!ring[idx].has_value()) break;
                    BasicHybridResultU8& r = *ring[idx];

                    Timer tw;
                    if (cfg.large.write_vector_bucket) {
                        if (!writer->AppendBlockRawU8(r.start, r.cluster_id, r.B_small, r.a,
                                                      r.x_u8.data.data(), /*ld_x_u8=*/d, err)) {
                            Fail(err && !err->empty() ? *err : "EncodeBaseStreaming(hybrid,u8): writer failed.");
                            return false;
                        }
                    }
                    else {
                        if (!writer->AppendBlock(r.start, r.cluster_id, r.B_small, r.a,
                                                 /*x_u8_optional=*/nullptr, err)) {
                            Fail(err && !err->empty() ? *err : "EncodeBaseStreaming(hybrid,u8): writer failed.");
                            return false;
                        }
                    }
                    if (!CommitBlock(r.start + static_cast<std::uint64_t>(r.cluster_id.size()))) {
                        Fail(err && !err->empty() ? *err : "EncodeBaseStreaming(hybrid,u8): commit hook failed.");
                        return false;
                    }
                    t_write_total += tw.ElapsedSeconds();

                    t_rotate_total += r.t.rotate;
                    t_copy_total += r.t.copy;
                    t_xc_total += r.t.xc;
                    t_beam_total += r.t.beam;
                    t_beam_root_gemm_total += r.t.beam_root_gemm;
                    t_beam_root_update_total += r.t.beam_root_update;
                    t_beam_expand_total += r.t.beam_expand;
                    t_ls_total += r.t.ls;
                    t_icm_total += r.t.icm;
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                    if (cfg.large.profile_timing) {
                        icm_cuda_total.upload_fixed += r.icm_cuda.upload_fixed;
                        icm_cuda_total.upload_B += r.icm_cuda.upload_B;
                        icm_cuda_total.icm_init += r.icm_cuda.icm_init;
                        icm_cuda_total.ils_copy_perturb += r.icm_cuda.ils_copy_perturb;
                        icm_cuda_total.ils_icm += r.icm_cuda.ils_icm;
                        icm_cuda_total.ils_accept += r.icm_cuda.ils_accept;
                        icm_cuda_total.download += r.icm_cuda.download;
                    }
                    used_cuda_icm_any = used_cuda_icm_any || r.cuda_icm_flags.used_any;
                    used_cuda_icm_any_device_xc = used_cuda_icm_any_device_xc || r.cuda_icm_flags.used_device_xc;
                    used_cuda_icm_any_device_x = used_cuda_icm_any_device_x || r.cuda_icm_flags.used_device_x;
#endif

                    if (want_mse && r.metrics.n > 0) {
                        mse_sum_beam += r.metrics.mse_sum_beam;
                        mse_sum_final += r.metrics.mse_sum_final;
                        n_total += r.metrics.n;
                    }

                    ring[idx].reset();
                    ++next_block;
                    ++blocks_written;
                    --inflight;
                    std::string msg = "Encoded block " + std::to_string(blocks_written) + " / " + std::to_string(
                        blocks_total);
                    if (cfg.large.profile_timing) {
                        msg += " (hybrid=1)";
                    }
                    if (blocks_written == blocks_total) {
                        LogInfoProgressDone(msg);
                    }
                    else {
                        LogInfoProgress(msg);
                    }
                }
                return true;
            };

            for (std::uint64_t start = start_id, bid = first_block; start < nbase; start += block, ++bid) {
                while (!failed.load() && inflight >= static_cast<std::uint64_t>(depth)) {
                    if (!DrainOneResultBlocking()) break;
                }
                if (failed.load()) break;

                const std::uint64_t remaining = nbase - start;
                const auto nread = static_cast<std::uint32_t>(std::min<std::uint64_t>(remaining, block));

                ColMajorMatrix<std::uint8_t> x_u8;
                PrefetchedBasicBlock<std::uint8_t> pref;
                Timer tr;
                if (async_io) {
                    if (!async_io->Pop(&pref)) {
                        const std::string msg = async_io->error();
                        Fail(msg.empty() ? "EncodeBaseStreaming(hybrid,u8): async basic IO pop failed." : msg);
                        break;
                    }
                    x_u8 = std::move(pref.X);
                }
                else {
                    std::string local_err;
                    if (!base_reader.ReadBlock(start, nread, &x_u8, &local_err)) {
                        Fail(local_err.empty() ? "EncodeBaseStreaming(hybrid,u8): ReadBlock failed." : local_err);
                        break;
                    }
                }
                t_read_total += tr.ElapsedSeconds();

                BasicHybridTaskU8 task;
                task.block_id = bid;
                task.start = start;
                task.nread = nread;
                task.x_u8 = std::move(x_u8);

                const bool to_cpu = (static_cast<int>(bid % static_cast<std::uint64_t>(cpu_stride)) == cpu_phase);
                if (to_cpu) {
                    if (!q_cpu.Push(std::move(task))) break;
                }
                else {
                    if (!q_gpu.Push(std::move(task))) break;
                }
                ++inflight;

                // Opportunistically drain already-completed results (non-blocking).
                BasicHybridResultU8 ready;
                while (q_res.TryPop(&ready)) {
                    const std::uint64_t rid = ready.block_id;
                    const std::uint64_t max_bid = next_block + static_cast<std::uint64_t>(depth) - 1;
                    if (rid < next_block || rid > max_bid) {
                        Fail("EncodeBaseStreaming(hybrid,u8): reorder depth too small (block_id out of window).");
                        break;
                    }
                    const auto slot = static_cast<std::size_t>(rid % static_cast<std::uint64_t>(depth));
                    if (ring[slot].has_value()) {
                        Fail("EncodeBaseStreaming(hybrid,u8): reorder ring slot collision (increase reorder depth).");
                        break;
                    }
                    ring[slot] = std::move(ready);
                    // Try to advance writer if possible.
                    while (next_block < blocks_total) {
                        const auto idx = static_cast<std::size_t>(next_block % static_cast<std::uint64_t>(depth));
                        if (!ring[idx].has_value()) break;
                        BasicHybridResultU8& r = *ring[idx];
                        Timer tw;
                        if (cfg.large.write_vector_bucket) {
                            if (!writer->AppendBlockRawU8(r.start, r.cluster_id, r.B_small, r.a,
                                                          r.x_u8.data.data(), /*ld_x_u8=*/d, err)) {
                                Fail(err && !err->empty() ? *err : "EncodeBaseStreaming(hybrid,u8): writer failed.");
                                break;
                            }
                        }
                        else {
                            if (!writer->AppendBlock(r.start, r.cluster_id, r.B_small, r.a,
                                                     /*x_u8_optional=*/nullptr, err)) {
                                Fail(err && !err->empty() ? *err : "EncodeBaseStreaming(hybrid,u8): writer failed.");
                                break;
                            }
                        }
                        if (!CommitBlock(r.start + static_cast<std::uint64_t>(r.cluster_id.size()))) {
                            Fail(err && !err->empty() ? *err : "EncodeBaseStreaming(hybrid,u8): commit hook failed.");
                            break;
                        }
                        t_write_total += tw.ElapsedSeconds();

                        t_rotate_total += r.t.rotate;
                        t_copy_total += r.t.copy;
                        t_xc_total += r.t.xc;
                        t_beam_total += r.t.beam;
                        t_beam_root_gemm_total += r.t.beam_root_gemm;
                        t_beam_root_update_total += r.t.beam_root_update;
                        t_beam_expand_total += r.t.beam_expand;
                        t_ls_total += r.t.ls;
                        t_icm_total += r.t.icm;
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                        if (cfg.large.profile_timing) {
                            icm_cuda_total.upload_fixed += r.icm_cuda.upload_fixed;
                            icm_cuda_total.upload_B += r.icm_cuda.upload_B;
                            icm_cuda_total.icm_init += r.icm_cuda.icm_init;
                            icm_cuda_total.ils_copy_perturb += r.icm_cuda.ils_copy_perturb;
                            icm_cuda_total.ils_icm += r.icm_cuda.ils_icm;
                            icm_cuda_total.ils_accept += r.icm_cuda.ils_accept;
                            icm_cuda_total.download += r.icm_cuda.download;
                        }
                        used_cuda_icm_any = used_cuda_icm_any || r.cuda_icm_flags.used_any;
                        used_cuda_icm_any_device_xc = used_cuda_icm_any_device_xc || r.cuda_icm_flags.used_device_xc;
                        used_cuda_icm_any_device_x = used_cuda_icm_any_device_x || r.cuda_icm_flags.used_device_x;
#endif
                        if (want_mse && r.metrics.n > 0) {
                            mse_sum_beam += r.metrics.mse_sum_beam;
                            mse_sum_final += r.metrics.mse_sum_final;
                            n_total += r.metrics.n;
                        }

                        ring[idx].reset();
                        ++next_block;
                        ++blocks_written;
                        --inflight;
                        std::string msg =
                            "Encoded block " + std::to_string(blocks_written) + " / " + std::to_string(blocks_total);
                        if (cfg.large.profile_timing) {
                            msg += " (hybrid=1)";
                        }
                        if (blocks_written == blocks_total) {
                            LogInfoProgressDone(msg);
                        }
                        else {
                            LogInfoProgress(msg);
                        }
                    }
                    if (failed.load()) break;
                }
                if (failed.load()) break;
            }

            if (async_io) {
                async_io->Stop();
            }
            q_gpu.Close();
            q_cpu.Close();

            // Drain remaining results.
            while (!failed.load() && blocks_written < blocks_total) {
                if (!DrainOneResultBlocking()) break;
            }

            q_res.Close();
            if (gpu_worker.joinable()) gpu_worker.join();
            if (cpu_worker.joinable()) cpu_worker.join();

            if (failed.load()) {
                if (err) {
                    std::lock_guard<std::mutex> guard(err_mu);
                    *err = worker_err.empty() ? "EncodeBaseStreaming(hybrid,u8): failed." : worker_err;
                }
                return false;
            }

            const float beam_mse = (want_mse && n_total > 0)
                                       ? static_cast<float>(mse_sum_beam / static_cast<double>(n_total))
                                       : 0.0f;
            const float final_mse = (want_mse && n_total > 0)
                                        ? static_cast<float>(mse_sum_final / static_cast<double>(n_total))
                                        : 0.0f;
            if (out_beam_mse) *out_beam_mse = beam_mse;
            if (out_final_mse) *out_final_mse = final_mse;
            if (cfg.train.log_metrics) {
                LogInfo("Base streaming beam MSE: " + FormatFloat(beam_mse, 6));
                LogInfo("Base streaming final MSE: " + FormatFloat(final_mse, 6));
            }

            if (cfg.large.profile_timing) {
                LogInfo("Base streaming total timing(s): read=" + FormatFloat(static_cast<float>(t_read_total), 3) +
                    " rotate=" + FormatFloat(static_cast<float>(t_rotate_total), 3) +
                    " copy=" + FormatFloat(static_cast<float>(t_copy_total), 3) +
                    " xC=" + FormatFloat(static_cast<float>(t_xc_total), 3) +
                    " beam=" + FormatFloat(static_cast<float>(t_beam_total), 3) +
                    (use_large_root
                         ? (" (root_gemm=" + FormatFloat(static_cast<float>(t_beam_root_gemm_total), 3) +
                             " root_upd=" + FormatFloat(static_cast<float>(t_beam_root_update_total), 3) +
                             " expand=" + FormatFloat(static_cast<float>(t_beam_expand_total), 3) + ")")
                         : "") +
                    " ls=" + FormatFloat(static_cast<float>(t_ls_total), 3) +
                    " icm_ils=" + FormatFloat(static_cast<float>(t_icm_total), 3) +
                    " write=" + FormatFloat(static_cast<float>(t_write_total), 3));
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                LogInfo("Base streaming total icm_cuda timing(s): up_fixed=" +
                    FormatFloat(static_cast<float>(icm_cuda_total.upload_fixed), 3) +
                    " up_B=" + FormatFloat(static_cast<float>(icm_cuda_total.upload_B), 3) +
                    " icm_init=" + FormatFloat(static_cast<float>(icm_cuda_total.icm_init), 3) +
                    " ils_copy=" + FormatFloat(static_cast<float>(icm_cuda_total.ils_copy_perturb), 3) +
                    " ils_icm=" + FormatFloat(static_cast<float>(icm_cuda_total.ils_icm), 3) +
                    " ils_accept=" + FormatFloat(static_cast<float>(icm_cuda_total.ils_accept), 3) +
                    " down=" + FormatFloat(static_cast<float>(icm_cuda_total.download), 3));
#endif
            }
            g_base_icm_backend_once.MaybeLog(
                /*f32_prefix=*/false,
                               used_cuda_icm_any
                                   ? ("Base streaming ICM backend: cuda=1 device_xC=" +
                                       std::to_string(used_cuda_icm_any_device_xc ? 1 : 0) +
                                       " device_X=" + std::to_string(used_cuda_icm_any_device_x ? 1 : 0))
                                   : "Base streaming ICM backend: cuda=0 (cpu)");
            LogInfo("Base streaming wall time(s): " + FormatFloat(static_cast<float>(t_wall.ElapsedSeconds()), 3));
            return true;
        }
        // Internal compute tile to bound xC_small memory.
        const int tile_n = 8192;

        BaseTileBuf buf;
        buf.Init(d, tile_n, pre_full.H, use_large_root ? pre_large->H_small : 0, m,
                 /*need_full_precomp=*/!use_large_root,
                 /*need_large_root=*/use_large_root);

        double t_read_total = 0.0;
        double t_rotate_total = 0.0;
        double t_copy_total = 0.0;
        double t_xc_total = 0.0;
        double t_beam_total = 0.0;
        double t_beam_root_gemm_total = 0.0;
        double t_beam_root_update_total = 0.0;
        double t_beam_expand_total = 0.0;
        double t_ls_total = 0.0;
        double t_icm_total = 0.0;
        double t_write_total = 0.0;
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
        CudaIcmTiming icm_cuda_total;
#endif
        bool used_cuda_icm_any = false;
        bool used_cuda_icm_any_device_xc = false;
        bool used_cuda_icm_any_device_x = false;

        double mse_sum_beam = 0.0;
        double mse_sum_final = 0.0;
        std::uint64_t n_total = 0;

        Timer t_wall;
        const std::uint64_t blocks_total = (nbase + block - 1) / block;

        std::unique_ptr<AsyncBasicBlockPrefetcher<io::BvecsReader, std::uint8_t>> async_io;
        if (cfg.runtime.basic_async_io && cfg.runtime.basic_async_io_depth > 0) {
            async_io = std::make_unique<AsyncBasicBlockPrefetcher<io::BvecsReader, std::uint8_t>>(
                base_reader, start_id, nbase, d, block,
                cfg.runtime.basic_async_io_depth,
                cfg.runtime.basic_async_io_mb);
            async_io->Start();
        }

        std::unique_ptr<AsyncBasicTileWriterU8> async_writer;
        const bool async_write_enabled =
        (cfg.runtime.basic_async_write &&
            cfg.runtime.basic_async_write_depth > 0 &&
            can_use_gpu_lane &&
            !want_commit_hook);
        if (async_write_enabled) {
            async_writer = std::make_unique<AsyncBasicTileWriterU8>(
                writer, d,
                /*write_raw_bucket=*/cfg.large.write_vector_bucket,
                cfg.runtime.basic_async_write_depth,
                cfg.runtime.basic_async_write_mb,
                m,
                tile_n,
                err);
        }

        for (std::uint64_t start = start_id; start < nbase; start += block) {
            Timer t_block;
            double t_read = 0.0, t_rot = 0.0, t_copy = 0.0, t_xc = 0.0, t_beam = 0.0, t_ls = 0.0, t_icm = 0.0, t_write =
                       0.0;
            double t_beam_root_gemm = 0.0, t_beam_root_update = 0.0, t_beam_expand = 0.0;
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
            CudaIcmTiming icm_cuda_block;
            bool used_cuda_icm = false;
#endif

            const std::uint64_t remaining = nbase - start;
            const auto nread = static_cast<std::uint32_t>(std::min<std::uint64_t>(remaining, block));

            ColMajorMatrix<std::uint8_t> x_u8;
            PrefetchedBasicBlock<std::uint8_t> pref;
            Timer t;
            if (async_io) {
                if (!async_io->Pop(&pref)) {
                    if (err) {
                        const std::string msg = async_io->error();
                        *err = msg.empty() ? "EncodeBaseStreaming: async basic IO pop failed." : msg;
                    }
                    return false;
                }
                x_u8 = std::move(pref.X);
            }
            else {
                if (!base_reader.ReadBlock(start, nread, &x_u8, err)) {
                    return false;
                }
            }
            t_read = t.ElapsedSeconds();

            // Process in compute tiles to avoid allocating huge xC_small.
            for (int t0 = 0; t0 < x_u8.cols; t0 += tile_n) {
                const int tlen = std::min(tile_n, x_u8.cols - t0);

                // Convert + rotate tile directly into Xblk (avoids allocating Xrot and the memcpy).
                buf.SetLen(tlen, /*need_full_precomp=*/!use_large_root, /*need_large_root=*/use_large_root);
                ColMajorMatrix<float>& Xtmp = buf.Xtmp;
                ColMajorMatrix<float>& Xblk = buf.Xblk;
                t.Reset();
                const std::uint8_t* src_u8 =
                    x_u8.data.data() + static_cast<std::size_t>(t0) * static_cast<std::size_t>(d);
                if (!kernels->IsGpu()) {
                    ConvertU8TileToF32(src_u8, d, tlen, Xtmp.data.data());
                    {
                        ScopedBlasThreads blas_scope(OmpMaxThreads());
                        GemmRaw(false, false,
                                d, tlen, d,
                                1.0f,
                                R.data.data(), R.rows,
                                Xtmp.data.data(), Xtmp.rows,
                                0.0f,
                                Xblk.data.data(), Xblk.rows);
                    }
                }
                else {
                    // GPU path: feed the raw u8 tile pointer directly (avoid host memcpy of the tile).
                    kernels->ConvertU8ToF32AndRotatePtr(src_u8, /*ld_in=*/d, /*rows=*/d, /*cols=*/tlen, R, &Xblk);
                }
                t_rot += t.ElapsedSeconds();

                // NOTE: cluster_id must have exact size when calling writer->AppendBlock.
                // Keep capacity and resize only when required.
                std::vector<std::uint32_t>& cluster_id = buf.cluster_id;
                ColMajorMatrix<Code>& B_small = buf.B_small;
                ColMajorMatrix<float>& a = buf.a;

                float beam_mse_block = 0.0f;
                float final_mse_block = 0.0f;

                if (!use_large_root) {
                    // Full precomp path (h0 small).
                    ColMajorMatrix<float>& xC = buf.xC_full;
                    t.Reset();
                    kernels->Gemm(true, false, 1.0f, pre_full.C_all, Xblk, 0.0f, &xC);
                    t_xc += t.ElapsedSeconds();

                    ColMajorMatrix<FullCode>& B_full = buf.B_full;
                    const int H_beam = std::max(1, cfg.base.encode.H_beam);
                    t.Reset();
                    BeamSearchPrefixLS(Xblk, pre_full, xC, H_beam, &B_full);
                    t_beam += t.ElapsedSeconds();

                    t.Reset();
                    ExtractClusterIdAndCodesSmall(B_full, &cluster_id, &B_small);
                    t_beam += t.ElapsedSeconds(); // tiny, but keep in beam stage

                    t.Reset();
                    SolveLeastSquaresAll(pre_full, xC, B_full, &a);
                    t_ls += t.ElapsedSeconds();
                    std::vector<float>& X_norm2 = buf.X_norm2;
                    std::vector<float>& cost_full = buf.cost_beam;
                    std::vector<float>* cost_full_ptr = want_mse ? &cost_full : nullptr;
                    if (want_mse) {
                        ComputeXNorm2(Xblk, &X_norm2);
                        ComputeCosts(pre_full, xC, B_full, a, X_norm2, &cost_full);
                        beam_mse_block = MeanMseFromCosts(cost_full);
                    }
                    else {
                        ComputeXNorm2(Xblk, &X_norm2);
                    }
                    const std::uint64_t sample_id_offset = start + static_cast<std::uint64_t>(t0);
                    t.Reset();
                    [[maybe_unused]] const bool want_cuda_icm =
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                        (cfg.runtime.use_cuda && kernels && kernels->IsGpu());
#else
                    false;
#endif
                    if (cfg.base.encode.use_abs) {
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                        if (want_cuda_icm) {
                            CudaIcmTiming icm_t;
                            DynamicIcmWithIlsAbsNoNormalCuda(Xblk, pre_full, xC,
                                                             cfg.base.encode.icm_iters,
                                                             cfg.base.encode.ils_iters,
                                                             cfg.base.encode.perturb_k,
                                                             static_cast<std::uint32_t>(cfg.base.encode.seed),
                                                             &B_full, &a, &X_norm2, cost_full_ptr,
                                                             /*print_progress=*/false,
                                                             sample_id_offset,
                                                             cfg.large.profile_timing ? &icm_t : nullptr);
                            if (cfg.large.profile_timing) {
                                used_cuda_icm = true;
                                icm_cuda_block.upload_fixed += icm_t.upload_fixed;
                                icm_cuda_block.upload_B += icm_t.upload_B;
                                icm_cuda_block.icm_init += icm_t.icm_init;
                                icm_cuda_block.ils_copy_perturb += icm_t.ils_copy_perturb;
                                icm_cuda_block.ils_icm += icm_t.ils_icm;
                                icm_cuda_block.ils_accept += icm_t.ils_accept;
                                icm_cuda_block.download += icm_t.download;
                            }
                        }
                        else
#endif
                        {
                            DynamicIcmWithIlsAbsNoNormal(Xblk, pre_full, xC,
                                                         cfg.base.encode.icm_iters,
                                                         cfg.base.encode.ils_iters,
                                                         cfg.base.encode.perturb_k,
                                                         static_cast<std::uint32_t>(cfg.base.encode.seed),
                                                         &B_full, &a, &X_norm2, cost_full_ptr,
                                                         /*print_progress=*/false,
                                                         sample_id_offset);
                        }
                    }
                    else {
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                        if (want_cuda_icm) {
                            CudaIcmTiming icm_t;
                            DynamicIcmWithIlsNoAbsNoNormalCuda(Xblk, pre_full, xC,
                                                               cfg.base.encode.icm_iters,
                                                               cfg.base.encode.ils_iters,
                                                               cfg.base.encode.perturb_k,
                                                               static_cast<std::uint32_t>(cfg.base.encode.seed),
                                                               &B_full, &a, &X_norm2, cost_full_ptr,
                                                               /*print_progress=*/false,
                                                               sample_id_offset,
                                                               cfg.large.profile_timing ? &icm_t : nullptr);
                            if (cfg.large.profile_timing) {
                                used_cuda_icm = true;
                                icm_cuda_block.upload_fixed += icm_t.upload_fixed;
                                icm_cuda_block.upload_B += icm_t.upload_B;
                                icm_cuda_block.icm_init += icm_t.icm_init;
                                icm_cuda_block.ils_copy_perturb += icm_t.ils_copy_perturb;
                                icm_cuda_block.ils_icm += icm_t.ils_icm;
                                icm_cuda_block.ils_accept += icm_t.ils_accept;
                                icm_cuda_block.download += icm_t.download;
                            }
                        }
                        else
#endif
                        {
                            DynamicIcmWithIlsNoAbsNoNormal(Xblk, pre_full, xC,
                                                           cfg.base.encode.icm_iters,
                                                           cfg.base.encode.ils_iters,
                                                           cfg.base.encode.perturb_k,
                                                           static_cast<std::uint32_t>(cfg.base.encode.seed),
                                                           &B_full, &a, &X_norm2, cost_full_ptr,
                                                           /*print_progress=*/false,
                                                           sample_id_offset);
                        }
                    }
                    t_icm += t.ElapsedSeconds();
                    if (want_mse) {
                        final_mse_block = MeanMseFromCosts(cost_full);
                    }

                    // Update cluster_id/B_small after ICM if codes changed.
                    ExtractClusterIdAndCodesSmall(B_full, &cluster_id, &B_small);
                }
                else {
                    // Large-root path: keep root code as cluster_id, store only small codes.
                    ColMajorMatrix<float>& xC_small = buf.xC_small;
                    t.Reset();
                    DeviceMatF32View xC_dev{};
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                    const bool want_cuda_beam_xc =
                    (cfg.runtime.use_cuda &&
                        kernels && kernels->IsGpu() &&
                        (cfg.base.encode.H_beam == 2 || cfg.base.encode.H_beam == 4));
                    if (want_cuda_beam_xc) {
                        kernels->GemmDevice(true, false, 1.0f, pre_large->C_small, Xblk, 0.0f, &xC_dev);
                    }
                    else
#endif
                    {
                        kernels->Gemm(true, false, 1.0f, pre_large->C_small, Xblk, 0.0f, &xC_small);
                    }
                    t_xc += t.ElapsedSeconds();
                    const int H_beam = std::max(1, cfg.base.encode.H_beam);
                    t.Reset();
                    BeamSearchLargeRootTiming beam_lr;
                    const bool want_cuda_icm_lr =
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                        (cfg.runtime.use_cuda && kernels && kernels->IsGpu());
#else
                    false;
#endif
                    const bool want_gpu_beam_mse = want_mse && want_cuda_icm_lr && xC_dev.ptr;
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                    const bool allow_tf32_beam = EffectiveCudaAllowTf32(cfg.runtime);
                    if (cfg.runtime.use_cuda) {
                        if (H_beam == 2) {
                            if (xC_dev.ptr) {
                                DeviceMatF32View X_dev{};
                                const bool have_X_dev = kernels && kernels->TryGetCachedDeviceF32(Xblk, &X_dev);
                                if (want_gpu_beam_mse) {
                                    if (have_X_dev) {
                                        BeamSearchPrefixLSLargeRootCudaH2DeviceXcDeviceXWithBestErr(
                                            X_dev, *pre_large, xC_dev,
                                            allow_tf32_beam,
                                            &cluster_id, &B_small,
                                            &buf.cost_beam,
                                            cfg.large.profile_timing ? &beam_lr : nullptr);
                                    }
                                    else {
                                        BeamSearchPrefixLSLargeRootCudaH2DeviceXcWithBestErr(
                                            Xblk, *pre_large, xC_dev,
                                            allow_tf32_beam,
                                            &cluster_id, &B_small,
                                            &buf.cost_beam,
                                            cfg.large.profile_timing ? &beam_lr : nullptr);
                                    }
                                }
                                else {
                                    if (have_X_dev) {
                                        BeamSearchPrefixLSLargeRootCudaH2DeviceXcDeviceX(
                                            X_dev, *pre_large, xC_dev,
                                            allow_tf32_beam,
                                            &cluster_id, &B_small,
                                            cfg.large.profile_timing ? &beam_lr : nullptr);
                                    }
                                    else {
                                        BeamSearchPrefixLSLargeRootCudaH2DeviceXc(Xblk, *pre_large, xC_dev,
                                            allow_tf32_beam,
                                            &cluster_id, &B_small,
                                            cfg.large.profile_timing ? &beam_lr : nullptr);
                                    }
                                }
                            }
                            else {
                                BeamSearchPrefixLSLargeRootCudaH2(Xblk, *pre_large, xC_small,
                                                                  allow_tf32_beam,
                                                                  &cluster_id, &B_small,
                                                                  cfg.large.profile_timing ? &beam_lr : nullptr);
                            }
                        }
                        else if (H_beam == 4) {
                            if (xC_dev.ptr) {
                                DeviceMatF32View X_dev{};
                                const bool have_X_dev = kernels && kernels->TryGetCachedDeviceF32(Xblk, &X_dev);
                                if (want_gpu_beam_mse) {
                                    if (have_X_dev) {
                                        BeamSearchPrefixLSLargeRootCudaH4DeviceXcDeviceXWithBestErr(
                                            X_dev, *pre_large, xC_dev,
                                            allow_tf32_beam,
                                            &cluster_id, &B_small,
                                            &buf.cost_beam,
                                            cfg.large.profile_timing ? &beam_lr : nullptr);
                                    }
                                    else {
                                        BeamSearchPrefixLSLargeRootCudaH4DeviceXcWithBestErr(
                                            Xblk, *pre_large, xC_dev,
                                            allow_tf32_beam,
                                            &cluster_id, &B_small,
                                            &buf.cost_beam,
                                            cfg.large.profile_timing ? &beam_lr : nullptr);
                                    }
                                }
                                else {
                                    if (have_X_dev) {
                                        BeamSearchPrefixLSLargeRootCudaH4DeviceXcDeviceX(
                                            X_dev, *pre_large, xC_dev,
                                            allow_tf32_beam,
                                            &cluster_id, &B_small,
                                            cfg.large.profile_timing ? &beam_lr : nullptr);
                                    }
                                    else {
                                        BeamSearchPrefixLSLargeRootCudaH4DeviceXc(Xblk, *pre_large, xC_dev,
                                            allow_tf32_beam,
                                            &cluster_id, &B_small,
                                            cfg.large.profile_timing ? &beam_lr : nullptr);
                                    }
                                }
                            }
                            else {
                                BeamSearchPrefixLSLargeRootCudaH4(Xblk, *pre_large, xC_small,
                                                                  allow_tf32_beam,
                                                                  &cluster_id, &B_small,
                                                                  cfg.large.profile_timing ? &beam_lr : nullptr);
                            }
                        }
                        else {
                            BeamSearchPrefixLSLargeRootCuda(Xblk, *pre_large, xC_small, H_beam,
                                                            allow_tf32_beam,
                                                            &cluster_id, &B_small,
                                                            cfg.large.profile_timing ? &beam_lr : nullptr);
                        }
                    }
                    else
#endif
                    {
                        BeamSearchPrefixLSLargeRoot(Xblk, *pre_large, xC_small, H_beam, &cluster_id, &B_small,
                                                    cfg.large.profile_timing ? &beam_lr : nullptr);
                    }
                    t_beam += t.ElapsedSeconds();
                    if (cfg.large.profile_timing) {
                        t_beam_root_gemm += beam_lr.root_gemm;
                        t_beam_root_update += beam_lr.root_update;
                        t_beam_expand += beam_lr.expand;
                    }

                    // If we computed xC_small on device, download it once after beam (beam consumed the device view).
                    // If we computed xC_small on device, download it only when the host path needs it
                    // (e.g. CPU LS / CPU MSE metrics). Large-root CUDA ICM can consume the device view directly.
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                    if (xC_dev.ptr && (!want_cuda_icm_lr || (want_mse && !want_gpu_beam_mse))) {
                        t.Reset();
                        kernels->DownloadF32(xC_dev, &xC_small);
                        t_xc += t.ElapsedSeconds();
                    }
#endif

                    // CPU LS is only needed for CPU ICM or for logging MSE metrics on the host.
                    if (!want_cuda_icm_lr || (want_mse && !want_gpu_beam_mse)) {
                        t.Reset();
                        SolveLeastSquaresAllLargeRoot(*pre_large, Xblk, xC_small, cluster_id, B_small, &a);
                        t_ls += t.ElapsedSeconds();
                    }

                    std::vector<float>& X_norm2 = buf.X_norm2;
                    std::vector<float>& cost_beam = buf.cost_beam;
                    if (want_mse) {
                        if (want_gpu_beam_mse) {
                            // cost_beam is filled by the CUDA beam search (best beam error per sample).
                            beam_mse_block = MeanMseFromCosts(cost_beam);
                        }
                        else {
                            ComputeXNorm2(Xblk, &X_norm2);
                            ComputeCostsLargeRoot(*pre_large, Xblk, xC_small, cluster_id, B_small, a,
                                                  X_norm2, &cost_beam);
                            beam_mse_block = MeanMseFromCosts(cost_beam);
                        }
                    }

                    if (!want_mse || want_gpu_beam_mse) {
                        ComputeXNorm2(Xblk, &X_norm2);
                    }
                    std::vector<float>* cost_final_ptr = want_mse ? &buf.cost_beam : nullptr;
                    const std::uint64_t sample_id_offset = start + static_cast<std::uint64_t>(t0);
                    t.Reset();
                    if (cfg.base.encode.use_abs) {
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                        if (want_cuda_icm_lr) {
                            CudaIcmTiming icm_t;
                            if (xC_dev.ptr) {
                                DeviceMatF32View X_dev{};
                                if (kernels->TryGetCachedDeviceF32(Xblk, &X_dev)) {
                                    used_cuda_icm_any = true;
                                    used_cuda_icm_any_device_xc = true;
                                    used_cuda_icm_any_device_x = true;
                                    DynamicIcmWithIlsAbsNoNormalLargeRootCudaDeviceXcDeviceX(X_dev, *pre_large, xC_dev,
                                        cfg.base.encode.icm_iters,
                                        cfg.base.encode.ils_iters,
                                        cfg.base.encode.perturb_k,
                                        static_cast<std::uint32_t>(cfg.base.encode.seed),
                                        cluster_id,
                                        &B_small, &a, &X_norm2, cost_final_ptr,
                                        /*print_progress=*/false,
                                        sample_id_offset,
                                        cfg.large.profile_timing ? &icm_t : nullptr);
                                }
                                else {
                                    used_cuda_icm_any = true;
                                    used_cuda_icm_any_device_xc = true;
                                    DynamicIcmWithIlsAbsNoNormalLargeRootCudaDeviceXc(Xblk, *pre_large, xC_dev,
                                        cfg.base.encode.icm_iters,
                                        cfg.base.encode.ils_iters,
                                        cfg.base.encode.perturb_k,
                                        static_cast<std::uint32_t>(cfg.base.encode.seed),
                                        cluster_id,
                                        &B_small, &a, &X_norm2, cost_final_ptr,
                                        /*print_progress=*/false,
                                        sample_id_offset,
                                        cfg.large.profile_timing ? &icm_t : nullptr);
                                }
                            }
                            else {
                                used_cuda_icm_any = true;
                                DynamicIcmWithIlsAbsNoNormalLargeRootCuda(Xblk, *pre_large, xC_small,
                                                                          cfg.base.encode.icm_iters,
                                                                          cfg.base.encode.ils_iters,
                                                                          cfg.base.encode.perturb_k,
                                                                          static_cast<std::uint32_t>(cfg.base.encode.
                                                                              seed),
                                                                          cluster_id,
                                                                          &B_small, &a, &X_norm2, cost_final_ptr,
                                                                          /*print_progress=*/false,
                                                                          sample_id_offset,
                                                                          cfg.large.profile_timing ? &icm_t : nullptr);
                            }
                            if (cfg.large.profile_timing) {
                                used_cuda_icm = true;
                                icm_cuda_block.upload_fixed += icm_t.upload_fixed;
                                icm_cuda_block.upload_B += icm_t.upload_B;
                                icm_cuda_block.icm_init += icm_t.icm_init;
                                icm_cuda_block.ils_copy_perturb += icm_t.ils_copy_perturb;
                                icm_cuda_block.ils_icm += icm_t.ils_icm;
                                icm_cuda_block.ils_accept += icm_t.ils_accept;
                                icm_cuda_block.download += icm_t.download;
                            }
                        }
                        else
#endif
                        {
                            DynamicIcmWithIlsAbsNoNormalLargeRoot(Xblk, *pre_large, xC_small,
                                                                  cfg.base.encode.icm_iters,
                                                                  cfg.base.encode.ils_iters,
                                                                  cfg.base.encode.perturb_k,
                                                                  static_cast<std::uint32_t>(cfg.base.encode.seed),
                                                                  cluster_id,
                                                                  &B_small, &a, &X_norm2, cost_final_ptr,
                                                                  /*print_progress=*/false,
                                                                  sample_id_offset);
                        }
                    }
                    else {
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                        if (want_cuda_icm_lr) {
                            CudaIcmTiming icm_t;
                            if (xC_dev.ptr) {
                                DeviceMatF32View X_dev{};
                                if (kernels->TryGetCachedDeviceF32(Xblk, &X_dev)) {
                                    used_cuda_icm_any = true;
                                    used_cuda_icm_any_device_xc = true;
                                    used_cuda_icm_any_device_x = true;
                                    DynamicIcmWithIlsNoAbsNoNormalLargeRootCudaDeviceXcDeviceX(
                                        X_dev, *pre_large, xC_dev,
                                        cfg.base.encode.icm_iters,
                                        cfg.base.encode.ils_iters,
                                        cfg.base.encode.perturb_k,
                                        static_cast<std::uint32_t>(cfg.base.encode.seed),
                                        cluster_id,
                                        &B_small, &a, &X_norm2, cost_final_ptr,
                                        /*print_progress=*/false,
                                        sample_id_offset,
                                        cfg.large.profile_timing ? &icm_t : nullptr);
                                }
                                else {
                                    used_cuda_icm_any = true;
                                    used_cuda_icm_any_device_xc = true;
                                    DynamicIcmWithIlsNoAbsNoNormalLargeRootCudaDeviceXc(Xblk, *pre_large, xC_dev,
                                        cfg.base.encode.icm_iters,
                                        cfg.base.encode.ils_iters,
                                        cfg.base.encode.perturb_k,
                                        static_cast<std::uint32_t>(cfg.base.encode.seed),
                                        cluster_id,
                                        &B_small, &a, &X_norm2, cost_final_ptr,
                                        /*print_progress=*/false,
                                        sample_id_offset,
                                        cfg.large.profile_timing ? &icm_t : nullptr);
                                }
                            }
                            else {
                                used_cuda_icm_any = true;
                                DynamicIcmWithIlsNoAbsNoNormalLargeRootCuda(Xblk, *pre_large, xC_small,
                                                                            cfg.base.encode.icm_iters,
                                                                            cfg.base.encode.ils_iters,
                                                                            cfg.base.encode.perturb_k,
                                                                            static_cast<std::uint32_t>(cfg.base.encode.
                                                                                seed),
                                                                            cluster_id,
                                                                            &B_small, &a, &X_norm2, cost_final_ptr,
                                                                            /*print_progress=*/false,
                                                                            sample_id_offset,
                                                                            cfg.large.profile_timing
                                                                                ? &icm_t
                                                                                : nullptr);
                            }
                            if (cfg.large.profile_timing) {
                                used_cuda_icm = true;
                                icm_cuda_block.upload_fixed += icm_t.upload_fixed;
                                icm_cuda_block.upload_B += icm_t.upload_B;
                                icm_cuda_block.icm_init += icm_t.icm_init;
                                icm_cuda_block.ils_copy_perturb += icm_t.ils_copy_perturb;
                                icm_cuda_block.ils_icm += icm_t.ils_icm;
                                icm_cuda_block.ils_accept += icm_t.ils_accept;
                                icm_cuda_block.download += icm_t.download;
                            }
                        }
                        else
#endif
                        {
                            DynamicIcmWithIlsNoAbsNoNormalLargeRoot(Xblk, *pre_large, xC_small,
                                                                    cfg.base.encode.icm_iters,
                                                                    cfg.base.encode.ils_iters,
                                                                    cfg.base.encode.perturb_k,
                                                                    static_cast<std::uint32_t>(cfg.base.encode.seed),
                                                                    cluster_id,
                                                                    &B_small, &a, &X_norm2, cost_final_ptr,
                                                                    /*print_progress=*/false,
                                                                    sample_id_offset);
                        }
                    }
                    t_icm += t.ElapsedSeconds();
                    if (want_mse) {
                        final_mse_block = MeanMseFromCosts(*cost_final_ptr);
                    }
                }

                if (want_mse) {
                    mse_sum_beam += static_cast<double>(beam_mse_block) * static_cast<double>(tlen);
                    mse_sum_final += static_cast<double>(final_mse_block) * static_cast<double>(tlen);
                    n_total += static_cast<std::uint64_t>(tlen);
                }

                // Write store block (global start_id uses base global ids).
                const std::uint64_t global_start = start + static_cast<std::uint64_t>(t0);
                t.Reset();
                if (async_writer && async_writer->Enabled()) {
                    BasicWriteTileJobU8 job;
                    job.global_start = global_start;
                    job.tlen = tlen;

                    job.cluster_id.resize(static_cast<std::size_t>(tlen));
                    std::memcpy(job.cluster_id.data(),
                                cluster_id.data(),
                                sizeof(std::uint32_t) * static_cast<std::size_t>(tlen));

                    const int rows_B = B_small.rows;
                    job.B_small.rows = rows_B;
                    job.B_small.cols = tlen;
                    job.B_small.data.resize(static_cast<std::size_t>(rows_B) * static_cast<std::size_t>(tlen));
                    std::memcpy(job.B_small.data.data(),
                                B_small.data.data(),
                                sizeof(Code) * static_cast<std::size_t>(rows_B) * static_cast<std::size_t>(tlen));

                    const int rows_a = a.rows;
                    job.a.rows = rows_a;
                    job.a.cols = tlen;
                    job.a.data.resize(static_cast<std::size_t>(rows_a) * static_cast<std::size_t>(tlen));
                    std::memcpy(job.a.data.data(),
                                a.data.data(),
                                sizeof(float) * static_cast<std::size_t>(rows_a) * static_cast<std::size_t>(tlen));

                    if (cfg.large.write_vector_bucket) {
                        job.raw_u8.resize(static_cast<std::size_t>(d) * static_cast<std::size_t>(tlen));
                        std::memcpy(job.raw_u8.data(),
                                    src_u8,
                                    static_cast<std::size_t>(d) * static_cast<std::size_t>(tlen));
                    }

                    if (!async_writer->Push(std::move(job))) {
                        return false;
                    }
                }
                else {
                    cluster_id.resize(static_cast<std::size_t>(tlen));
                    if (cfg.large.write_vector_bucket) {
                        if (!writer->AppendBlockRawU8(global_start,
                                                      cluster_id,
                                                      B_small,
                                                      a,
                                                      src_u8,
                                                      /*ld_x_u8=*/d,
                                                      err)) {
                            return false;
                        }
                    }
                    else {
                        if (!writer->AppendBlock(global_start,
                                                 cluster_id,
                                                 B_small,
                                                 a,
                                                 /*x_u8_optional=*/nullptr,
                                                 err)) {
                            return false;
                        }
                    }
                }
                t_write += t.ElapsedSeconds();
            }

            t_read_total += t_read;
            t_rotate_total += t_rot;
            t_copy_total += t_copy;
            t_xc_total += t_xc;
            t_beam_total += t_beam;
            t_beam_root_gemm_total += t_beam_root_gemm;
            t_beam_root_update_total += t_beam_root_update;
            t_beam_expand_total += t_beam_expand;
            t_ls_total += t_ls;
            t_icm_total += t_icm;
            t_write_total += t_write;
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
            if (cfg.large.profile_timing && used_cuda_icm) {
                icm_cuda_total.upload_fixed += icm_cuda_block.upload_fixed;
                icm_cuda_total.upload_B += icm_cuda_block.upload_B;
                icm_cuda_total.icm_init += icm_cuda_block.icm_init;
                icm_cuda_total.ils_copy_perturb += icm_cuda_block.ils_copy_perturb;
                icm_cuda_total.ils_icm += icm_cuda_block.ils_icm;
                icm_cuda_total.ils_accept += icm_cuda_block.ils_accept;
                icm_cuda_total.download += icm_cuda_block.download;
            }
#endif

            if (cfg.large.profile_timing) {
                LogInfo("  block " + std::to_string(start / block + 1) + " / " + std::to_string(blocks_total) +
                    " timing(s): read=" + FormatFloat(static_cast<float>(t_read), 4) +
                    " rotate=" + FormatFloat(static_cast<float>(t_rot), 4) +
                    " copy=" + FormatFloat(static_cast<float>(t_copy), 4) +
                    " xC=" + FormatFloat(static_cast<float>(t_xc), 4) +
                    " beam=" + FormatFloat(static_cast<float>(t_beam), 4) +
                    (use_large_root
                         ? (" (root_gemm=" + FormatFloat(static_cast<float>(t_beam_root_gemm), 4) +
                             " root_upd=" + FormatFloat(static_cast<float>(t_beam_root_update), 4) +
                             " expand=" + FormatFloat(static_cast<float>(t_beam_expand), 4) + ")")
                         : "") +
                    " ls=" + FormatFloat(static_cast<float>(t_ls), 4) +
                    " icm_ils=" + FormatFloat(static_cast<float>(t_icm), 4) +
                    " write=" + FormatFloat(static_cast<float>(t_write), 4) +
                    " total=" + FormatFloat(static_cast<float>(t_block.ElapsedSeconds()), 4));
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                if (used_cuda_icm) {
                    LogInfo("  icm_cuda timing(s): up_fixed=" + FormatFloat(
                            static_cast<float>(icm_cuda_block.upload_fixed), 4) +
                        " up_B=" + FormatFloat(static_cast<float>(icm_cuda_block.upload_B), 4) +
                        " icm_init=" + FormatFloat(static_cast<float>(icm_cuda_block.icm_init), 4) +
                        " ils_copy=" + FormatFloat(static_cast<float>(icm_cuda_block.ils_copy_perturb), 4) +
                        " ils_icm=" + FormatFloat(static_cast<float>(icm_cuda_block.ils_icm), 4) +
                        " ils_accept=" + FormatFloat(static_cast<float>(icm_cuda_block.ils_accept), 4) +
                        " down=" + FormatFloat(static_cast<float>(icm_cuda_block.download), 4));
                }
#endif
            }

            const bool last_block = (start + block >= nbase);
            const std::uint64_t block_id = start / block;
            std::string msg = "Encoded block " + std::to_string(block_id + 1) + " / " + std::to_string(blocks_total);
            if (cfg.large.profile_timing) {
                msg += " (block_total_s=" + FormatFloat(static_cast<float>(t_block.ElapsedSeconds()), 4) + ")";
            }
            if (last_block) {
                LogInfoProgressDone(msg);
            }
            else {
                LogInfoProgress(msg);
            }

            // Commit checkpoint only at block boundaries (after all tiles in the block are written).
            const std::uint64_t next_id = start + static_cast<std::uint64_t>(nread);
            if (!CommitBlock(next_id)) {
                return false;
            }
        }

        if (async_io) {
            async_io->Stop();
        }
        if (async_writer && async_writer->Enabled()) {
            Timer t_flush;
            if (!async_writer->Finish()) {
                return false;
            }
            t_write_total += t_flush.ElapsedSeconds();
        }

        const float beam_mse = (want_mse && n_total > 0)
                                   ? static_cast<float>(mse_sum_beam / static_cast<double>(n_total))
                                   : 0.0f;
        const float final_mse = (want_mse && n_total > 0)
                                    ? static_cast<float>(mse_sum_final / static_cast<double>(n_total))
                                    : 0.0f;
        if (out_beam_mse) *out_beam_mse = beam_mse;
        if (out_final_mse) *out_final_mse = final_mse;
        if (cfg.train.log_metrics) {
            LogInfo("Base streaming beam MSE: " + FormatFloat(beam_mse, 6));
            LogInfo("Base streaming final MSE: " + FormatFloat(final_mse, 6));
        }
        if (cfg.large.profile_timing) {
            LogInfo("Base streaming total timing(s): read=" + FormatFloat(static_cast<float>(t_read_total), 3) +
                " rotate=" + FormatFloat(static_cast<float>(t_rotate_total), 3) +
                " copy=" + FormatFloat(static_cast<float>(t_copy_total), 3) +
                " xC=" + FormatFloat(static_cast<float>(t_xc_total), 3) +
                " beam=" + FormatFloat(static_cast<float>(t_beam_total), 3) +
                (use_large_root
                     ? (" (root_gemm=" + FormatFloat(static_cast<float>(t_beam_root_gemm_total), 3) +
                         " root_upd=" + FormatFloat(static_cast<float>(t_beam_root_update_total), 3) +
                         " expand=" + FormatFloat(static_cast<float>(t_beam_expand_total), 3) + ")")
                     : "") +
                " ls=" + FormatFloat(static_cast<float>(t_ls_total), 3) +
                " icm_ils=" + FormatFloat(static_cast<float>(t_icm_total), 3) +
                " write=" + FormatFloat(static_cast<float>(t_write_total), 3));
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
            LogInfo("Base streaming total icm_cuda timing(s): up_fixed=" +
                FormatFloat(static_cast<float>(icm_cuda_total.upload_fixed), 3) +
                " up_B=" + FormatFloat(static_cast<float>(icm_cuda_total.upload_B), 3) +
                " icm_init=" + FormatFloat(static_cast<float>(icm_cuda_total.icm_init), 3) +
                " ils_copy=" + FormatFloat(static_cast<float>(icm_cuda_total.ils_copy_perturb), 3) +
                " ils_icm=" + FormatFloat(static_cast<float>(icm_cuda_total.ils_icm), 3) +
                " ils_accept=" + FormatFloat(static_cast<float>(icm_cuda_total.ils_accept), 3) +
                " down=" + FormatFloat(static_cast<float>(icm_cuda_total.download), 3));
#endif
        }
        g_base_icm_backend_once.MaybeLog(
            /*f32_prefix=*/false,
                           used_cuda_icm_any
                               ? ("Base streaming ICM backend: cuda=1 device_xC=" +
                                   std::to_string(used_cuda_icm_any_device_xc ? 1 : 0) +
                                   " device_X=" + std::to_string(used_cuda_icm_any_device_x ? 1 : 0))
                               : "Base streaming ICM backend: cuda=0 (cpu)");
        LogInfo("Base streaming wall time(s): " + FormatFloat(static_cast<float>(t_wall.ElapsedSeconds()), 3));
        return true;
    }

    template <typename ReaderF32T>
    static bool EncodeBaseStreamingF32Impl(const Config& cfg,
                                           const ReaderF32T& base_reader,
                                           const ColMajorMatrix<float>& R,
                                           const CodebookPack& C_root,
                                           const Precomp& pre_full,
                                           const PrecompLargeRoot* pre_large,
                                           StreamKernelProvider* kernels,
                                           io::BaseBasicWriter* writer,
                                           float* out_beam_mse,
                                           float* out_final_mse,
                                           std::string* err,
                                           std::uint64_t start_id,
                                           const StreamingBasicCommitHook* commit_hook) {
        if (!writer) {
            if (err) *err = "EncodeBaseStreamingF32: writer is null.";
            return false;
        }
        std::uint64_t nbase = base_reader.n();
        if (cfg.dataset.nbase_set && cfg.dataset.nbase > 0) {
            nbase = std::min<std::uint64_t>(nbase, static_cast<std::uint64_t>(cfg.dataset.nbase));
        }
        if (start_id > nbase) {
            if (err) *err = "EncodeBaseStreamingF32: start_id out of range.";
            return false;
        }
        const int d = base_reader.d();
        if (d <= 0 || nbase == 0) {
            if (err) *err = "EncodeBaseStreamingF32: empty base reader.";
            return false;
        }
        if (R.rows != d || R.cols != d) {
            if (err) *err = "EncodeBaseStreamingF32: rotation matrix size mismatch.";
            return false;
        }
        const int m = cfg.model.m;
        if (m <= 1) {
            if (err) *err = "EncodeBaseStreamingF32: model.m must be >= 2.";
            return false;
        }

        const bool use_large_root = (pre_large && pre_large->ready && pre_large->h_vec.size() == C_root.books.size());

        CpuStreamKernels cpu_kernels;
        if (!kernels) {
            kernels = &cpu_kernels;
        }

        const bool can_use_gpu_lane =
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
            (cfg.runtime.use_cuda && kernels && kernels->IsGpu());
#else
        false;
#endif
        const bool use_hybrid =
        (cfg.runtime.basic_hybrid_enable &&
            cfg.runtime.basic_hybrid_cpu_stride > 0 &&
            can_use_gpu_lane);

        // Avoid re-printing the same header logs for each global R-iteration.
        {
            static std::uint64_t last_tag = 0;
            static bool printed = false;
            std::uint64_t tag = 1469598103934665603ULL;
            auto Mix = [&](std::uint64_t v)
            {
                tag ^= v;
                tag *= 1099511628211ULL;
            };
            Mix(nbase);
            Mix(static_cast<std::uint64_t>(d));
            Mix(static_cast<std::uint64_t>(m));
            Mix(static_cast<std::uint64_t>(std::max(1, cfg.base.encode.H_beam)));
            Mix(static_cast<std::uint64_t>(std::max(0, cfg.base.encode.icm_iters)));
            Mix(static_cast<std::uint64_t>(std::max(0, cfg.base.encode.ils_iters)));
            Mix(static_cast<std::uint64_t>(std::max(0, cfg.base.encode.perturb_k)));
            Mix(static_cast<std::uint64_t>(cfg.base.encode.use_abs ? 1 : 0));
            Mix(static_cast<std::uint64_t>(cfg.base.encode.seed));
            Mix(static_cast<std::uint64_t>(std::max(1, cfg.large.base_block)));
            Mix(static_cast<std::uint64_t>(use_large_root ? 1 : 0));
            Mix(static_cast<std::uint64_t>(cfg.large.profile_timing ? 1 : 0));
            Mix(static_cast<std::uint64_t>(use_hybrid ? 1 : 0));
            if (use_hybrid) {
                Mix(static_cast<std::uint64_t>(std::max(0, cfg.runtime.basic_hybrid_cpu_stride)));
                Mix(static_cast<std::uint64_t>(std::max(0, cfg.runtime.basic_hybrid_cpu_threads)));
                Mix(static_cast<std::uint64_t>(std::max(0, cfg.runtime.basic_hybrid_reorder_depth)));
                Mix(static_cast<std::uint64_t>(std::max(0, cfg.runtime.basic_hybrid_inflight_mb)));
            }
            if (!printed || tag != last_tag) {
                printed = true;
                last_tag = tag;
                LogInfo("Base streaming(F32) encode params: nbase=" + std::to_string(nbase) +
                    " d=" + std::to_string(d) +
                    " m=" + std::to_string(m) +
                    " H_beam=" + std::to_string(std::max(1, cfg.base.encode.H_beam)) +
                    " icm_iters=" + std::to_string(std::max(0, cfg.base.encode.icm_iters)) +
                    " ils_iters=" + std::to_string(std::max(0, cfg.base.encode.ils_iters)) +
                    " perturb_k=" + std::to_string(std::max(0, cfg.base.encode.perturb_k)) +
                    " use_abs=" + std::string(cfg.base.encode.use_abs ? "true" : "false") +
                    " seed=" + std::to_string(cfg.base.encode.seed));
                LogInfo("Base streaming(F32) IO: base_block=" + std::to_string(std::max(1, cfg.large.base_block)) +
                    " tile_n=8192" +
                    " write_vector_bucket=" + std::string(cfg.large.write_vector_bucket ? "true" : "false") +
                    " write_basic_to_bucket=" + std::string(cfg.large.write_basic_to_bucket ? "true" : "false") +
                    " profile_timing=" + std::string(cfg.large.profile_timing ? "true" : "false"));
                if (use_hybrid) {
                    const int cpu_threads =
                        (cfg.runtime.basic_hybrid_cpu_threads > 0)
                            ? cfg.runtime.basic_hybrid_cpu_threads
                            : OmpMaxThreads();
                    LogInfo("Base streaming(F32) hybrid: enabled (cpu_stride=" + std::to_string(
                            cfg.runtime.basic_hybrid_cpu_stride) +
                        " cpu_threads=" + std::to_string(cpu_threads) +
                        " reorder_depth=" + std::to_string(std::max(1, cfg.runtime.basic_hybrid_reorder_depth)) +
                        " inflight_mb=" + std::to_string(std::max(0, cfg.runtime.basic_hybrid_inflight_mb)) + ")");
                }
                else if (cfg.runtime.basic_hybrid_enable && cfg.runtime.basic_hybrid_cpu_stride > 0) {
                    LogInfo(
                        "Base streaming(F32) hybrid: requested but disabled (requires runtime.use_cuda=true and CUDA stream kernels).");
                }
                if (use_large_root) {
                    LogInfo("Base streaming(F32) path: large-root (h0=" + std::to_string(pre_large->h_vec[0]) +
                        " H_small=" + std::to_string(pre_large->H_small) + ")");
                }
                else {
                    LogInfo("Base streaming(F32) path: full-precomp (H=" + std::to_string(pre_full.H) + ")");
                }
            }
        }
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
        CudaIcmTiming icm_cuda_total;
#endif

        const std::uint64_t block = static_cast<std::uint64_t>(std::max(1, cfg.large.base_block));

        const bool want_commit_hook = (commit_hook && commit_hook->OnCommit);
        if (want_commit_hook && block > 0 && (start_id % block) != 0) {
            if (err)
                *err =
                    "EncodeBaseStreamingF32: start_id must be a multiple of large.base_block when using checkpointing.";
            return false;
        }
        auto CommitBlock = [&](std::uint64_t next_id) -> bool
        {
            if (!want_commit_hook) return true;
            if (!writer->Flush(err)) return false;
            return commit_hook->OnCommit(next_id, writer, commit_hook->ctx, err);
        };

        // MSE reporting is optional; disable it to reduce overhead when metrics are not needed.
        const bool want_mse = cfg.train.log_metrics || (out_beam_mse != nullptr) || (out_final_mse != nullptr);

        if (use_hybrid) {
            const int m_codes = std::max(0, m - 1);
            std::uint64_t bytes_per_vec = sizeof(std::uint32_t) +
                static_cast<std::uint64_t>(m_codes) * sizeof(Code) +
                static_cast<std::uint64_t>(m) * sizeof(float);
            if (cfg.large.write_vector_bucket) {
                bytes_per_vec += static_cast<std::uint64_t>(d) * sizeof(float); // raw f32
            }
            const std::uint64_t bytes_per_block = bytes_per_vec * block;
            int depth = std::max(2, cfg.runtime.basic_hybrid_reorder_depth);
            depth = ClampDepthByBudget(depth, cfg.runtime.basic_hybrid_inflight_mb, bytes_per_block);
            if (depth < 2) depth = 2;

            const int cpu_stride = std::max(1, cfg.runtime.basic_hybrid_cpu_stride);
            const int cpu_phase = cpu_stride - 1;
            const int cpu_threads =
                (cfg.runtime.basic_hybrid_cpu_threads > 0) ? cfg.runtime.basic_hybrid_cpu_threads : OmpMaxThreads();

            CpuStreamKernels cpu_lane_kernels;
            BoundedQueue<BasicHybridTaskF32> q_gpu(depth);
            BoundedQueue<BasicHybridTaskF32> q_cpu(depth);
            BoundedQueue<BasicHybridResultF32> q_res(depth);

            std::atomic<bool> failed{false};
            std::mutex err_mu;
            std::string worker_err;
            auto Fail = [&](const std::string& msg)
            {
                failed.store(true);
                {
                    std::lock_guard<std::mutex> guard(err_mu);
                    if (worker_err.empty()) worker_err = msg;
                }
                q_gpu.Close();
                q_cpu.Close();
                q_res.Close();
            };

            std::thread gpu_worker([&]()
            {
                ScopedOmpThreads omp_scope(1);
                BasicHybridTaskF32 task;
                while (!failed.load() && q_gpu.Pop(&task)) {
                    BasicHybridResultF32 res;
                    std::string local_err;
                    if (!EncodeBasicHybridBlockF32(cfg, R, pre_full, pre_large, use_large_root, kernels, want_mse,
                                                   &task, &res, &local_err)) {
                        Fail(local_err.empty() ? "EncodeBaseStreamingF32(hybrid): GPU lane failed." : local_err);
                        break;
                    }
                    if (!q_res.Push(std::move(res))) break;
                }
            });

            std::thread cpu_worker([&]()
            {
                ScopedOmpThreads omp_scope(cpu_threads);
                BasicHybridTaskF32 task;
                while (!failed.load() && q_cpu.Pop(&task)) {
                    BasicHybridResultF32 res;
                    std::string local_err;
                    if (!EncodeBasicHybridBlockF32(cfg, R, pre_full, pre_large, use_large_root, &cpu_lane_kernels,
                                                   want_mse,
                                                   &task, &res, &local_err)) {
                        Fail(local_err.empty() ? "EncodeBaseStreamingF32(hybrid): CPU lane failed." : local_err);
                        break;
                    }
                    if (!q_res.Push(std::move(res))) break;
                }
            });

            std::vector<std::optional<BasicHybridResultF32>> ring(static_cast<std::size_t>(depth));
            const std::uint64_t first_block = (block > 0) ? (start_id / block) : 0;
            std::uint64_t next_block = first_block;
            std::uint64_t blocks_written = first_block;
            std::uint64_t inflight = 0;

            double t_read_total = 0.0;
            double t_write_total = 0.0;
            double t_rotate_total = 0.0;
            double t_copy_total = 0.0;
            double t_xc_total = 0.0;
            double t_beam_total = 0.0;
            double t_beam_root_gemm_total = 0.0;
            double t_beam_root_update_total = 0.0;
            double t_beam_expand_total = 0.0;
            double t_ls_total = 0.0;
            double t_icm_total = 0.0;
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
            CudaIcmTiming icm_cuda_total;
            bool used_cuda_icm_any = false;
            bool used_cuda_icm_any_device_xc = false;
            bool used_cuda_icm_any_device_x = false;
#else
        bool used_cuda_icm_any = false;
        bool used_cuda_icm_any_device_xc = false;
        bool used_cuda_icm_any_device_x = false;
#endif

            double mse_sum_beam = 0.0;
            double mse_sum_final = 0.0;
            std::uint64_t n_total = 0;

            Timer t_wall;
            const std::uint64_t blocks_total = (nbase + block - 1) / block;

            std::unique_ptr<AsyncBasicBlockPrefetcher<ReaderF32T, float>> async_io;
            if (cfg.runtime.basic_async_io && cfg.runtime.basic_async_io_depth > 0) {
                async_io = std::make_unique<AsyncBasicBlockPrefetcher<ReaderF32T, float>>(
                    base_reader, start_id, nbase, d, block,
                    cfg.runtime.basic_async_io_depth,
                    cfg.runtime.basic_async_io_mb);
                async_io->Start();
            }

            auto PlaceResult = [&](BasicHybridResultF32&& got) -> bool
            {
                const std::uint64_t bid = got.block_id;
                const std::uint64_t max_bid = next_block + static_cast<std::uint64_t>(depth) - 1;
                if (bid < next_block || bid > max_bid) {
                    Fail("EncodeBaseStreamingF32(hybrid): reorder depth too small (block_id out of window).");
                    return false;
                }
                const auto slot = static_cast<std::size_t>(bid % static_cast<std::uint64_t>(depth));
                if (ring[slot].has_value()) {
                    Fail("EncodeBaseStreamingF32(hybrid): reorder ring slot collision (increase reorder depth).");
                    return false;
                }
                ring[slot] = std::move(got);
                return true;
            };

            auto TryWriteReady = [&]() -> bool
            {
                while (next_block < blocks_total) {
                    const auto idx = static_cast<std::size_t>(next_block % static_cast<std::uint64_t>(depth));
                    if (!ring[idx].has_value()) break;
                    BasicHybridResultF32& r = *ring[idx];

                    Timer tw;
                    if (cfg.large.write_vector_bucket) {
                        if (!writer->AppendBlockRawF32(r.start, r.cluster_id, r.B_small, r.a,
                                                       r.x_f32.data.data(), /*ld_x_f32=*/d, err)) {
                            Fail(err && !err->empty() ? *err : "EncodeBaseStreamingF32(hybrid): writer failed.");
                            return false;
                        }
                    }
                    else {
                        if (!writer->AppendBlock(r.start, r.cluster_id, r.B_small, r.a,
                                                 /*x_u8_optional=*/nullptr, err)) {
                            Fail(err && !err->empty() ? *err : "EncodeBaseStreamingF32(hybrid): writer failed.");
                            return false;
                        }
                    }
                    if (!CommitBlock(r.start + static_cast<std::uint64_t>(r.cluster_id.size()))) {
                        Fail(err && !err->empty() ? *err : "EncodeBaseStreamingF32(hybrid): commit hook failed.");
                        return false;
                    }
                    t_write_total += tw.ElapsedSeconds();

                    t_rotate_total += r.t.rotate;
                    t_copy_total += r.t.copy;
                    t_xc_total += r.t.xc;
                    t_beam_total += r.t.beam;
                    t_beam_root_gemm_total += r.t.beam_root_gemm;
                    t_beam_root_update_total += r.t.beam_root_update;
                    t_beam_expand_total += r.t.beam_expand;
                    t_ls_total += r.t.ls;
                    t_icm_total += r.t.icm;
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                    if (cfg.large.profile_timing) {
                        icm_cuda_total.upload_fixed += r.icm_cuda.upload_fixed;
                        icm_cuda_total.upload_B += r.icm_cuda.upload_B;
                        icm_cuda_total.icm_init += r.icm_cuda.icm_init;
                        icm_cuda_total.ils_copy_perturb += r.icm_cuda.ils_copy_perturb;
                        icm_cuda_total.ils_icm += r.icm_cuda.ils_icm;
                        icm_cuda_total.ils_accept += r.icm_cuda.ils_accept;
                        icm_cuda_total.download += r.icm_cuda.download;
                    }
                    used_cuda_icm_any = used_cuda_icm_any || r.cuda_icm_flags.used_any;
                    used_cuda_icm_any_device_xc = used_cuda_icm_any_device_xc || r.cuda_icm_flags.used_device_xc;
                    used_cuda_icm_any_device_x = used_cuda_icm_any_device_x || r.cuda_icm_flags.used_device_x;
#endif
                    if (want_mse && r.metrics.n > 0) {
                        mse_sum_beam += r.metrics.mse_sum_beam;
                        mse_sum_final += r.metrics.mse_sum_final;
                        n_total += r.metrics.n;
                    }

                    ring[idx].reset();
                    ++next_block;
                    ++blocks_written;
                    --inflight;
                    std::string msg = "Encoded block " + std::to_string(blocks_written) + " / " + std::to_string(
                        blocks_total);
                    if (cfg.large.profile_timing) msg += " (hybrid=1)";
                    if (blocks_written == blocks_total) {
                        LogInfoProgressDone(msg);
                    }
                    else {
                        LogInfoProgress(msg);
                    }
                }
                return true;
            };

            auto DrainOneBlocking = [&]() -> bool
            {
                BasicHybridResultF32 got;
                if (!q_res.Pop(&got)) return false;
                if (!PlaceResult(std::move(got))) return false;
                return TryWriteReady();
            };

            for (std::uint64_t start = start_id, bid = first_block; start < nbase; start += block, ++bid) {
                while (!failed.load() && inflight >= static_cast<std::uint64_t>(depth)) {
                    if (!DrainOneBlocking()) break;
                }
                if (failed.load()) break;

                const std::uint64_t remaining = nbase - start;
                const auto nread = static_cast<std::uint32_t>(std::min<std::uint64_t>(remaining, block));

                ColMajorMatrix<float> Xf;
                PrefetchedBasicBlock<float> pref;
                Timer tr;
                if (async_io) {
                    if (!async_io->Pop(&pref)) {
                        const std::string msg = async_io->error();
                        Fail(msg.empty() ? "EncodeBaseStreamingF32(hybrid): async basic IO pop failed." : msg);
                        break;
                    }
                    Xf = std::move(pref.X);
                }
                else {
                    std::string local_err;
                    if (!base_reader.ReadBlock(start, nread, &Xf, &local_err)) {
                        Fail(local_err.empty() ? "EncodeBaseStreamingF32(hybrid): ReadBlock failed." : local_err);
                        break;
                    }
                }
                t_read_total += tr.ElapsedSeconds();

                BasicHybridTaskF32 task;
                task.block_id = bid;
                task.start = start;
                task.nread = nread;
                task.x_f32 = std::move(Xf);

                const bool to_cpu = (static_cast<int>(bid % static_cast<std::uint64_t>(cpu_stride)) == cpu_phase);
                if (to_cpu) {
                    if (!q_cpu.Push(std::move(task))) break;
                }
                else {
                    if (!q_gpu.Push(std::move(task))) break;
                }
                ++inflight;

                BasicHybridResultF32 ready;
                while (q_res.TryPop(&ready)) {
                    if (!PlaceResult(std::move(ready))) break;
                    if (!TryWriteReady()) break;
                    if (failed.load()) break;
                }
                if (failed.load()) break;
            }

            if (async_io) async_io->Stop();
            q_gpu.Close();
            q_cpu.Close();

            while (!failed.load() && blocks_written < blocks_total) {
                if (!DrainOneBlocking()) break;
            }

            q_res.Close();
            if (gpu_worker.joinable()) gpu_worker.join();
            if (cpu_worker.joinable()) cpu_worker.join();

            if (failed.load()) {
                if (err) {
                    std::lock_guard<std::mutex> guard(err_mu);
                    *err = worker_err.empty() ? "EncodeBaseStreamingF32(hybrid): failed." : worker_err;
                }
                return false;
            }

            const float beam_mse = (want_mse && n_total > 0)
                                       ? static_cast<float>(mse_sum_beam / static_cast<double>(n_total))
                                       : 0.0f;
            const float final_mse = (want_mse && n_total > 0)
                                        ? static_cast<float>(mse_sum_final / static_cast<double>(n_total))
                                        : 0.0f;
            if (out_beam_mse) *out_beam_mse = beam_mse;
            if (out_final_mse) *out_final_mse = final_mse;
            if (cfg.train.log_metrics) {
                LogInfo("Base streaming(F32) beam MSE: " + FormatFloat(beam_mse, 6));
                LogInfo("Base streaming(F32) final MSE: " + FormatFloat(final_mse, 6));
            }
            if (cfg.large.profile_timing) {
                LogInfo(
                    "Base streaming(F32) total timing(s): read=" + FormatFloat(static_cast<float>(t_read_total), 3) +
                    " rotate=" + FormatFloat(static_cast<float>(t_rotate_total), 3) +
                    " copy=" + FormatFloat(static_cast<float>(t_copy_total), 3) +
                    " xC=" + FormatFloat(static_cast<float>(t_xc_total), 3) +
                    " beam=" + FormatFloat(static_cast<float>(t_beam_total), 3) +
                    (use_large_root
                         ? (" (root_gemm=" + FormatFloat(static_cast<float>(t_beam_root_gemm_total), 3) +
                             " root_upd=" + FormatFloat(static_cast<float>(t_beam_root_update_total), 3) +
                             " expand=" + FormatFloat(static_cast<float>(t_beam_expand_total), 3) + ")")
                         : "") +
                    " ls=" + FormatFloat(static_cast<float>(t_ls_total), 3) +
                    " icm_ils=" + FormatFloat(static_cast<float>(t_icm_total), 3) +
                    " write=" + FormatFloat(static_cast<float>(t_write_total), 3));
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                LogInfo("Base streaming(F32) total icm_cuda timing(s): up_fixed=" +
                    FormatFloat(static_cast<float>(icm_cuda_total.upload_fixed), 3) +
                    " up_B=" + FormatFloat(static_cast<float>(icm_cuda_total.upload_B), 3) +
                    " icm_init=" + FormatFloat(static_cast<float>(icm_cuda_total.icm_init), 3) +
                    " ils_copy=" + FormatFloat(static_cast<float>(icm_cuda_total.ils_copy_perturb), 3) +
                    " ils_icm=" + FormatFloat(static_cast<float>(icm_cuda_total.ils_icm), 3) +
                    " ils_accept=" + FormatFloat(static_cast<float>(icm_cuda_total.ils_accept), 3) +
                    " down=" + FormatFloat(static_cast<float>(icm_cuda_total.download), 3));
#endif
            }
            g_base_icm_backend_once.MaybeLog(
                /*f32_prefix=*/true,
                               used_cuda_icm_any
                                   ? ("Base streaming(F32) ICM backend: cuda=1 device_xC=" +
                                       std::to_string(used_cuda_icm_any_device_xc ? 1 : 0) +
                                       " device_X=" + std::to_string(used_cuda_icm_any_device_x ? 1 : 0))
                                   : "Base streaming(F32) ICM backend: cuda=0 (cpu)");
            LogInfo("Base streaming(F32) wall time(s): " + FormatFloat(static_cast<float>(t_wall.ElapsedSeconds()), 3));
            return true;
        }

        const int tile_n = 8192;

        BaseTileBuf buf;
        buf.Init(d, tile_n, pre_full.H, use_large_root ? pre_large->H_small : 0, m,
                 /*need_full_precomp=*/!use_large_root,
                 /*need_large_root=*/use_large_root);

        double t_read_total = 0.0;
        double t_rotate_total = 0.0;
        double t_copy_total = 0.0;
        double t_xc_total = 0.0;
        double t_beam_total = 0.0;
        double t_beam_root_gemm_total = 0.0;
        double t_beam_root_update_total = 0.0;
        double t_beam_expand_total = 0.0;
        double t_ls_total = 0.0;
        double t_icm_total = 0.0;
        double t_write_total = 0.0;

        double mse_sum_beam = 0.0;
        double mse_sum_final = 0.0;
        std::uint64_t n_total = 0;
        Timer t_wall;
        const std::uint64_t blocks_total = (nbase + block - 1) / block;

        std::unique_ptr<AsyncBasicBlockPrefetcher<ReaderF32T, float>> async_io;
        if (cfg.runtime.basic_async_io && cfg.runtime.basic_async_io_depth > 0) {
            async_io = std::make_unique<AsyncBasicBlockPrefetcher<ReaderF32T, float>>(
                base_reader, start_id, nbase, d, block,
                cfg.runtime.basic_async_io_depth,
                cfg.runtime.basic_async_io_mb);
            async_io->Start();
        }

        std::unique_ptr<AsyncBasicTileWriterF32> async_writer;
        const bool async_write_enabled =
        (cfg.runtime.basic_async_write &&
            cfg.runtime.basic_async_write_depth > 0 &&
            can_use_gpu_lane &&
            !use_hybrid &&
            !want_commit_hook);
        if (async_write_enabled) {
            async_writer = std::make_unique<AsyncBasicTileWriterF32>(
                writer, d,
                /*write_raw_bucket=*/cfg.large.write_vector_bucket,
                cfg.runtime.basic_async_write_depth,
                cfg.runtime.basic_async_write_mb,
                m,
                tile_n,
                err);
        }

        for (std::uint64_t start = start_id; start < nbase; start += block) {
            Timer t_block;
            double t_read = 0.0, t_rot = 0.0, t_copy = 0.0, t_xc = 0.0, t_beam = 0.0, t_ls = 0.0, t_icm = 0.0, t_write =
                       0.0;
            double t_beam_root_gemm = 0.0, t_beam_root_update = 0.0, t_beam_expand = 0.0;

            const std::uint64_t remaining = nbase - start;
            const auto nread = static_cast<std::uint32_t>(std::min<std::uint64_t>(remaining, block));

#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
            CudaIcmTiming icm_cuda_block;
            bool used_cuda_icm = false;
#endif

            ColMajorMatrix<float> Xf;
            Timer t;
            if (async_io) {
                PrefetchedBasicBlock<float> pref;
                if (!async_io->Pop(&pref)) {
                    if (err) {
                        const std::string msg = async_io->error();
                        *err = msg.empty() ? "EncodeBaseStreamingF32: async basic IO pop failed." : msg;
                    }
                    return false;
                }
                Xf = std::move(pref.X);
            }
            else {
                if (!base_reader.ReadBlock(start, nread, &Xf, err)) {
                    return false;
                }
            }
            t_read = t.ElapsedSeconds();

            for (int t0 = 0; t0 < Xf.cols; t0 += tile_n) {
                const int tlen = std::min(tile_n, Xf.cols - t0);
                buf.SetLen(tlen, /*need_full_precomp=*/!use_large_root, /*need_large_root=*/use_large_root);
                ColMajorMatrix<float>& Xblk = buf.Xblk;
                t.Reset();
                const float* x_ptr =
                    Xf.data.data() + static_cast<std::size_t>(t0) * static_cast<std::size_t>(d);
                if (!kernels->IsGpu()) {
                    ScopedBlasThreads blas_scope(OmpMaxThreads());
                    GemmRaw(false, false,
                            d, tlen, d,
                            1.0f,
                            R.data.data(), R.rows,
                            x_ptr, d,
                            0.0f,
                            Xblk.data.data(), Xblk.rows);
                }
                else {
                    // GPU path: copy the tile into an owning ColMajorMatrix<float> and GEMM on device.
                    ColMajorMatrix<float>& Xtmp = buf.Xtmp;
                    Xtmp.rows = d;
                    Xtmp.cols = tlen;
                    std::memcpy(Xtmp.data.data(),
                                x_ptr,
                                sizeof(float) * static_cast<std::size_t>(d) * static_cast<std::size_t>(tlen));
                    kernels->Gemm(false, false, 1.0f, R, Xtmp, 0.0f, &Xblk);
                }
                t_rot += t.ElapsedSeconds();

                std::vector<std::uint32_t>& cluster_id = buf.cluster_id;
                ColMajorMatrix<Code>& B_small = buf.B_small;
                ColMajorMatrix<float>& a = buf.a;

                float beam_mse_block = 0.0f;
                float final_mse_block = 0.0f;

                if (!use_large_root) {
                    ColMajorMatrix<float>& xC = buf.xC_full;
                    [[maybe_unused]] DeviceMatF32View xC_dev{};
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                    [[maybe_unused]] const bool want_cuda_icm =
                        (cfg.runtime.use_cuda && kernels && kernels->IsGpu());
#else
                [[maybe_unused]] const bool want_cuda_icm = false;
#endif
                    t.Reset();
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                    if (want_cuda_icm) {
                        kernels->GemmDevice(true, false, 1.0f, pre_full.C_all, Xblk, 0.0f, &xC_dev);
                        kernels->DownloadF32(xC_dev, &xC);
                    }
                    else
#endif
                    {
                        kernels->Gemm(true, false, 1.0f, pre_full.C_all, Xblk, 0.0f, &xC);
                    }
                    t_xc += t.ElapsedSeconds();
                    ColMajorMatrix<FullCode>& B_full = buf.B_full;
                    const int H_beam = std::max(1, cfg.base.encode.H_beam);
                    t.Reset();
                    BeamSearchPrefixLS(Xblk, pre_full, xC, H_beam, &B_full);
                    t_beam += t.ElapsedSeconds();
                    ExtractClusterIdAndCodesSmall(B_full, &cluster_id, &B_small);
                    t.Reset();
                    SolveLeastSquaresAll(pre_full, xC, B_full, &a);
                    t_ls += t.ElapsedSeconds();

                    std::vector<float>& cost_beam = buf.cost_beam;
                    std::vector<float>& X_norm2 = buf.X_norm2;
                    if (want_mse) {
                        ComputeXNorm2(Xblk, &X_norm2);
                        ComputeCosts(pre_full, xC, B_full, a, X_norm2, &cost_beam);
                        beam_mse_block = MeanMseFromCosts(cost_beam);
                    }

                    if (!want_mse) {
                        ComputeXNorm2(Xblk, &X_norm2);
                    }
                    std::vector<float>& cost_full = buf.cost_beam;
                    std::vector<float>* cost_full_ptr = want_mse ? &cost_full : nullptr;
                    const std::uint64_t sample_id_offset = start + static_cast<std::uint64_t>(t0);
                    t.Reset();
                    if (cfg.base.encode.use_abs) {
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                        if (want_cuda_icm) {
                            CudaIcmTiming icm_t;
                            if (xC_dev.ptr) {
                                DynamicIcmWithIlsAbsNoNormalCudaDeviceXc(Xblk, pre_full, xC_dev,
                                                                         cfg.base.encode.icm_iters,
                                                                         cfg.base.encode.ils_iters,
                                                                         cfg.base.encode.perturb_k,
                                                                         static_cast<std::uint32_t>(cfg.base.encode.
                                                                             seed),
                                                                         &B_full, &a, &X_norm2, cost_full_ptr,
                                                                         /*print_progress=*/false,
                                                                         sample_id_offset,
                                                                         cfg.large.profile_timing ? &icm_t : nullptr);
                            }
                            else {
                                DynamicIcmWithIlsAbsNoNormalCuda(Xblk, pre_full, xC,
                                                                 cfg.base.encode.icm_iters,
                                                                 cfg.base.encode.ils_iters,
                                                                 cfg.base.encode.perturb_k,
                                                                 static_cast<std::uint32_t>(cfg.base.encode.seed),
                                                                 &B_full, &a, &X_norm2, cost_full_ptr,
                                                                 /*print_progress=*/false,
                                                                 sample_id_offset,
                                                                 cfg.large.profile_timing ? &icm_t : nullptr);
                            }
                            if (cfg.large.profile_timing) {
                                used_cuda_icm = true;
                                icm_cuda_block.upload_fixed += icm_t.upload_fixed;
                                icm_cuda_block.upload_B += icm_t.upload_B;
                                icm_cuda_block.icm_init += icm_t.icm_init;
                                icm_cuda_block.ils_copy_perturb += icm_t.ils_copy_perturb;
                                icm_cuda_block.ils_icm += icm_t.ils_icm;
                                icm_cuda_block.ils_accept += icm_t.ils_accept;
                                icm_cuda_block.download += icm_t.download;
                            }
                        }
                        else
#endif
                        {
                            DynamicIcmWithIlsAbsNoNormal(Xblk, pre_full, xC,
                                                         cfg.base.encode.icm_iters,
                                                         cfg.base.encode.ils_iters,
                                                         cfg.base.encode.perturb_k,
                                                         static_cast<std::uint32_t>(cfg.base.encode.seed),
                                                         &B_full, &a, &X_norm2, cost_full_ptr,
                                                         /*print_progress=*/false,
                                                         sample_id_offset);
                        }
                    }
                    else {
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                        if (want_cuda_icm) {
                            CudaIcmTiming icm_t;
                            if (xC_dev.ptr) {
                                DynamicIcmWithIlsNoAbsNoNormalCudaDeviceXc(Xblk, pre_full, xC_dev,
                                                                           cfg.base.encode.icm_iters,
                                                                           cfg.base.encode.ils_iters,
                                                                           cfg.base.encode.perturb_k,
                                                                           static_cast<std::uint32_t>(cfg.base.encode.
                                                                               seed),
                                                                           &B_full, &a, &X_norm2, cost_full_ptr,
                                                                           /*print_progress=*/false,
                                                                           sample_id_offset,
                                                                           cfg.large.profile_timing ? &icm_t : nullptr);
                            }
                            else {
                                DynamicIcmWithIlsNoAbsNoNormalCuda(Xblk, pre_full, xC,
                                                                   cfg.base.encode.icm_iters,
                                                                   cfg.base.encode.ils_iters,
                                                                   cfg.base.encode.perturb_k,
                                                                   static_cast<std::uint32_t>(cfg.base.encode.seed),
                                                                   &B_full, &a, &X_norm2, cost_full_ptr,
                                                                   /*print_progress=*/false,
                                                                   sample_id_offset,
                                                                   cfg.large.profile_timing ? &icm_t : nullptr);
                            }
                            if (cfg.large.profile_timing) {
                                used_cuda_icm = true;
                                icm_cuda_block.upload_fixed += icm_t.upload_fixed;
                                icm_cuda_block.upload_B += icm_t.upload_B;
                                icm_cuda_block.icm_init += icm_t.icm_init;
                                icm_cuda_block.ils_copy_perturb += icm_t.ils_copy_perturb;
                                icm_cuda_block.ils_icm += icm_t.ils_icm;
                                icm_cuda_block.ils_accept += icm_t.ils_accept;
                                icm_cuda_block.download += icm_t.download;
                            }
                        }
                        else
#endif
                        {
                            DynamicIcmWithIlsNoAbsNoNormal(Xblk, pre_full, xC,
                                                           cfg.base.encode.icm_iters,
                                                           cfg.base.encode.ils_iters,
                                                           cfg.base.encode.perturb_k,
                                                           static_cast<std::uint32_t>(cfg.base.encode.seed),
                                                           &B_full, &a, &X_norm2, cost_full_ptr,
                                                           /*print_progress=*/false,
                                                           sample_id_offset);
                        }
                    }
                    t_icm += t.ElapsedSeconds();
                    if (want_mse) {
                        final_mse_block = MeanMseFromCosts(cost_full);
                    }
                    ExtractClusterIdAndCodesSmall(B_full, &cluster_id, &B_small);
                }
                else {
                    ColMajorMatrix<float>& xC_small = buf.xC_small;
                    t.Reset();
                    DeviceMatF32View xC_dev{};
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                    const bool want_cuda_beam_xc =
                    (cfg.runtime.use_cuda &&
                        kernels && kernels->IsGpu() &&
                        (cfg.base.encode.H_beam == 2 || cfg.base.encode.H_beam == 4));
                    if (want_cuda_beam_xc) {
                        kernels->GemmDevice(true, false, 1.0f, pre_large->C_small, Xblk, 0.0f, &xC_dev);
                    }
                    else
#endif
                    {
                        kernels->Gemm(true, false, 1.0f, pre_large->C_small, Xblk, 0.0f, &xC_small);
                    }
                    t_xc += t.ElapsedSeconds();
                    const int H_beam = std::max(1, cfg.base.encode.H_beam);
                    t.Reset();
                    BeamSearchLargeRootTiming beam_lr;
                    const bool want_cuda_icm_lr =
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                        (cfg.runtime.use_cuda && kernels && kernels->IsGpu());
#else
                    false;
#endif
                    const bool want_gpu_beam_mse = want_mse && want_cuda_icm_lr && xC_dev.ptr;
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                    const bool allow_tf32_beam = EffectiveCudaAllowTf32(cfg.runtime);
                    if (cfg.runtime.use_cuda) {
                        if (H_beam == 2 && xC_dev.ptr) {
                            DeviceMatF32View X_dev{};
                            const bool have_X_dev = kernels && kernels->TryGetCachedDeviceF32(Xblk, &X_dev);
                            if (want_gpu_beam_mse) {
                                if (have_X_dev) {
                                    BeamSearchPrefixLSLargeRootCudaH2DeviceXcDeviceXWithBestErr(
                                        X_dev, *pre_large, xC_dev,
                                        allow_tf32_beam,
                                        &cluster_id, &B_small,
                                        &buf.cost_beam,
                                        cfg.large.profile_timing ? &beam_lr : nullptr);
                                }
                                else {
                                    BeamSearchPrefixLSLargeRootCudaH2DeviceXcWithBestErr(
                                        Xblk, *pre_large, xC_dev,
                                        allow_tf32_beam,
                                        &cluster_id, &B_small,
                                        &buf.cost_beam,
                                        cfg.large.profile_timing ? &beam_lr : nullptr);
                                }
                            }
                            else {
                                if (have_X_dev) {
                                    BeamSearchPrefixLSLargeRootCudaH2DeviceXcDeviceX(
                                        X_dev, *pre_large, xC_dev,
                                        allow_tf32_beam,
                                        &cluster_id, &B_small,
                                        cfg.large.profile_timing ? &beam_lr : nullptr);
                                }
                                else {
                                    BeamSearchPrefixLSLargeRootCudaH2DeviceXc(Xblk, *pre_large, xC_dev,
                                                                              allow_tf32_beam,
                                                                              &cluster_id, &B_small,
                                                                              cfg.large.profile_timing
                                                                                  ? &beam_lr
                                                                                  : nullptr);
                                }
                            }
                        }
                        else if (H_beam == 4 && xC_dev.ptr) {
                            DeviceMatF32View X_dev{};
                            const bool have_X_dev = kernels && kernels->TryGetCachedDeviceF32(Xblk, &X_dev);
                            if (want_gpu_beam_mse) {
                                if (have_X_dev) {
                                    BeamSearchPrefixLSLargeRootCudaH4DeviceXcDeviceXWithBestErr(
                                        X_dev, *pre_large, xC_dev,
                                        allow_tf32_beam,
                                        &cluster_id, &B_small,
                                        &buf.cost_beam,
                                        cfg.large.profile_timing ? &beam_lr : nullptr);
                                }
                                else {
                                    BeamSearchPrefixLSLargeRootCudaH4DeviceXcWithBestErr(
                                        Xblk, *pre_large, xC_dev,
                                        allow_tf32_beam,
                                        &cluster_id, &B_small,
                                        &buf.cost_beam,
                                        cfg.large.profile_timing ? &beam_lr : nullptr);
                                }
                            }
                            else {
                                if (have_X_dev) {
                                    BeamSearchPrefixLSLargeRootCudaH4DeviceXcDeviceX(
                                        X_dev, *pre_large, xC_dev,
                                        allow_tf32_beam,
                                        &cluster_id, &B_small,
                                        cfg.large.profile_timing ? &beam_lr : nullptr);
                                }
                                else {
                                    BeamSearchPrefixLSLargeRootCudaH4DeviceXc(Xblk, *pre_large, xC_dev,
                                                                              allow_tf32_beam,
                                                                              &cluster_id, &B_small,
                                                                              cfg.large.profile_timing
                                                                                  ? &beam_lr
                                                                                  : nullptr);
                                }
                            }
                        }
                        else {
                            BeamSearchPrefixLSLargeRootCuda(Xblk, *pre_large, xC_small, H_beam,
                                                            allow_tf32_beam,
                                                            &cluster_id, &B_small,
                                                            cfg.large.profile_timing ? &beam_lr : nullptr);
                        }
                    }
                    else
#endif
                    {
                        BeamSearchPrefixLSLargeRoot(Xblk, *pre_large, xC_small, H_beam, &cluster_id, &B_small,
                                                    cfg.large.profile_timing ? &beam_lr : nullptr);
                    }
                    t_beam += t.ElapsedSeconds();
                    if (cfg.large.profile_timing) {
                        t_beam_root_gemm += beam_lr.root_gemm;
                        t_beam_root_update += beam_lr.root_update;
                        t_beam_expand += beam_lr.expand;
                    }

                    // If we computed xC_small on device, download it only when the host path needs it
                    // (e.g. CPU LS / CPU MSE metrics). Large-root CUDA ICM can consume the device view directly.
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                    if (xC_dev.ptr && (!want_cuda_icm_lr || (want_mse && !want_gpu_beam_mse))) {
                        t.Reset();
                        kernels->DownloadF32(xC_dev, &xC_small);
                        t_xc += t.ElapsedSeconds();
                    }
#endif

                    // CPU LS is only needed for CPU ICM or for logging MSE metrics on the host.
                    if (!want_cuda_icm_lr || (want_mse && !want_gpu_beam_mse)) {
                        t.Reset();
                        SolveLeastSquaresAllLargeRoot(*pre_large, Xblk, xC_small, cluster_id, B_small, &a);
                        t_ls += t.ElapsedSeconds();
                    }

                    std::vector<float>& X_norm2 = buf.X_norm2;
                    std::vector<float>& cost_beam = buf.cost_beam;
                    if (want_mse) {
                        if (want_gpu_beam_mse) {
                            beam_mse_block = MeanMseFromCosts(cost_beam);
                        }
                        else {
                            ComputeXNorm2(Xblk, &X_norm2);
                            ComputeCostsLargeRoot(*pre_large, Xblk, xC_small, cluster_id, B_small, a,
                                                  X_norm2, &cost_beam);
                            beam_mse_block = MeanMseFromCosts(cost_beam);
                        }
                    }

                    if (!want_mse || want_gpu_beam_mse) {
                        ComputeXNorm2(Xblk, &X_norm2);
                    }
                    std::vector<float>* cost_final_ptr = want_mse ? &buf.cost_beam : nullptr;
                    const std::uint64_t sample_id_offset = start + static_cast<std::uint64_t>(t0);
                    t.Reset();
                    if (cfg.base.encode.use_abs) {
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                        if (want_cuda_icm_lr) {
                            CudaIcmTiming icm_t;
                            if (xC_dev.ptr) {
                                DeviceMatF32View X_dev{};
                                if (kernels->TryGetCachedDeviceF32(Xblk, &X_dev)) {
                                    DynamicIcmWithIlsAbsNoNormalLargeRootCudaDeviceXcDeviceX(X_dev, *pre_large, xC_dev,
                                        cfg.base.encode.icm_iters,
                                        cfg.base.encode.ils_iters,
                                        cfg.base.encode.perturb_k,
                                        static_cast<std::uint32_t>(cfg.base.encode.seed),
                                        cluster_id,
                                        &B_small, &a, &X_norm2, cost_final_ptr,
                                        /*print_progress=*/false,
                                        sample_id_offset,
                                        cfg.large.profile_timing ? &icm_t : nullptr);
                                }
                                else {
                                    DynamicIcmWithIlsAbsNoNormalLargeRootCudaDeviceXc(Xblk, *pre_large, xC_dev,
                                        cfg.base.encode.icm_iters,
                                        cfg.base.encode.ils_iters,
                                        cfg.base.encode.perturb_k,
                                        static_cast<std::uint32_t>(cfg.base.encode.seed),
                                        cluster_id,
                                        &B_small, &a, &X_norm2, cost_final_ptr,
                                        /*print_progress=*/false,
                                        sample_id_offset,
                                        cfg.large.profile_timing ? &icm_t : nullptr);
                                }
                            }
                            else {
                                DynamicIcmWithIlsAbsNoNormalLargeRootCuda(Xblk, *pre_large, xC_small,
                                                                          cfg.base.encode.icm_iters,
                                                                          cfg.base.encode.ils_iters,
                                                                          cfg.base.encode.perturb_k,
                                                                          static_cast<std::uint32_t>(cfg.base.encode.
                                                                              seed),
                                                                          cluster_id,
                                                                          &B_small, &a, &X_norm2, cost_final_ptr,
                                                                          /*print_progress=*/false,
                                                                          sample_id_offset,
                                                                          cfg.large.profile_timing ? &icm_t : nullptr);
                            }
                            if (cfg.large.profile_timing) {
                                used_cuda_icm = true;
                                icm_cuda_block.upload_fixed += icm_t.upload_fixed;
                                icm_cuda_block.upload_B += icm_t.upload_B;
                                icm_cuda_block.icm_init += icm_t.icm_init;
                                icm_cuda_block.ils_copy_perturb += icm_t.ils_copy_perturb;
                                icm_cuda_block.ils_icm += icm_t.ils_icm;
                                icm_cuda_block.ils_accept += icm_t.ils_accept;
                                icm_cuda_block.download += icm_t.download;
                            }
                        }
                        else
#endif
                        {
                            DynamicIcmWithIlsAbsNoNormalLargeRoot(Xblk, *pre_large, xC_small,
                                                                  cfg.base.encode.icm_iters,
                                                                  cfg.base.encode.ils_iters,
                                                                  cfg.base.encode.perturb_k,
                                                                  static_cast<std::uint32_t>(cfg.base.encode.seed),
                                                                  cluster_id,
                                                                  &B_small, &a, &X_norm2, cost_final_ptr,
                                                                  /*print_progress=*/false,
                                                                  sample_id_offset);
                        }
                    }
                    else {
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                        if (want_cuda_icm_lr) {
                            CudaIcmTiming icm_t;
                            if (xC_dev.ptr) {
                                DeviceMatF32View X_dev{};
                                if (kernels->TryGetCachedDeviceF32(Xblk, &X_dev)) {
                                    DynamicIcmWithIlsNoAbsNoNormalLargeRootCudaDeviceXcDeviceX(
                                        X_dev, *pre_large, xC_dev,
                                        cfg.base.encode.icm_iters,
                                        cfg.base.encode.ils_iters,
                                        cfg.base.encode.perturb_k,
                                        static_cast<std::uint32_t>(cfg.base.encode.seed),
                                        cluster_id,
                                        &B_small, &a, &X_norm2, cost_final_ptr,
                                        /*print_progress=*/false,
                                        sample_id_offset,
                                        cfg.large.profile_timing ? &icm_t : nullptr);
                                }
                                else {
                                    DynamicIcmWithIlsNoAbsNoNormalLargeRootCudaDeviceXc(Xblk, *pre_large, xC_dev,
                                        cfg.base.encode.icm_iters,
                                        cfg.base.encode.ils_iters,
                                        cfg.base.encode.perturb_k,
                                        static_cast<std::uint32_t>(cfg.base.encode.seed),
                                        cluster_id,
                                        &B_small, &a, &X_norm2, cost_final_ptr,
                                        /*print_progress=*/false,
                                        sample_id_offset,
                                        cfg.large.profile_timing ? &icm_t : nullptr);
                                }
                            }
                            else {
                                DynamicIcmWithIlsNoAbsNoNormalLargeRootCuda(Xblk, *pre_large, xC_small,
                                                                            cfg.base.encode.icm_iters,
                                                                            cfg.base.encode.ils_iters,
                                                                            cfg.base.encode.perturb_k,
                                                                            static_cast<std::uint32_t>(cfg.base.encode.
                                                                                seed),
                                                                            cluster_id,
                                                                            &B_small, &a, &X_norm2, cost_final_ptr,
                                                                            /*print_progress=*/false,
                                                                            sample_id_offset,
                                                                            cfg.large.profile_timing
                                                                                ? &icm_t
                                                                                : nullptr);
                            }
                            if (cfg.large.profile_timing) {
                                used_cuda_icm = true;
                                icm_cuda_block.upload_fixed += icm_t.upload_fixed;
                                icm_cuda_block.upload_B += icm_t.upload_B;
                                icm_cuda_block.icm_init += icm_t.icm_init;
                                icm_cuda_block.ils_copy_perturb += icm_t.ils_copy_perturb;
                                icm_cuda_block.ils_icm += icm_t.ils_icm;
                                icm_cuda_block.ils_accept += icm_t.ils_accept;
                                icm_cuda_block.download += icm_t.download;
                            }
                        }
                        else
#endif
                        {
                            DynamicIcmWithIlsNoAbsNoNormalLargeRoot(Xblk, *pre_large, xC_small,
                                                                    cfg.base.encode.icm_iters,
                                                                    cfg.base.encode.ils_iters,
                                                                    cfg.base.encode.perturb_k,
                                                                    static_cast<std::uint32_t>(cfg.base.encode.seed),
                                                                    cluster_id,
                                                                    &B_small, &a, &X_norm2, cost_final_ptr,
                                                                    /*print_progress=*/false,
                                                                    sample_id_offset);
                        }
                    }
                    t_icm += t.ElapsedSeconds();
                    if (want_mse) {
                        final_mse_block = MeanMseFromCosts(*cost_final_ptr);
                    }
                }

                if (want_mse) {
                    mse_sum_beam += static_cast<double>(beam_mse_block) * static_cast<double>(tlen);
                    mse_sum_final += static_cast<double>(final_mse_block) * static_cast<double>(tlen);
                    n_total += static_cast<std::uint64_t>(tlen);
                }

                const std::uint64_t global_start = start + static_cast<std::uint64_t>(t0);
                t.Reset();
                if (async_writer && async_writer->Enabled()) {
                    BasicWriteTileJobF32 job;
                    job.global_start = global_start;
                    job.tlen = tlen;

                    job.cluster_id.resize(static_cast<std::size_t>(tlen));
                    std::memcpy(job.cluster_id.data(),
                                cluster_id.data(),
                                sizeof(std::uint32_t) * static_cast<std::size_t>(tlen));

                    const int rows_B = B_small.rows;
                    job.B_small.rows = rows_B;
                    job.B_small.cols = tlen;
                    job.B_small.data.resize(static_cast<std::size_t>(rows_B) * static_cast<std::size_t>(tlen));
                    std::memcpy(job.B_small.data.data(),
                                B_small.data.data(),
                                sizeof(Code) * static_cast<std::size_t>(rows_B) * static_cast<std::size_t>(tlen));

                    const int rows_a = a.rows;
                    job.a.rows = rows_a;
                    job.a.cols = tlen;
                    job.a.data.resize(static_cast<std::size_t>(rows_a) * static_cast<std::size_t>(tlen));
                    std::memcpy(job.a.data.data(),
                                a.data.data(),
                                sizeof(float) * static_cast<std::size_t>(rows_a) * static_cast<std::size_t>(tlen));

                    if (cfg.large.write_vector_bucket) {
                        job.raw_f32.resize(static_cast<std::size_t>(d) * static_cast<std::size_t>(tlen));
                        std::memcpy(job.raw_f32.data(),
                                    x_ptr,
                                    sizeof(float) * static_cast<std::size_t>(d) * static_cast<std::size_t>(tlen));
                    }

                    if (!async_writer->Push(std::move(job))) {
                        return false;
                    }
                }
                else {
                    cluster_id.resize(static_cast<std::size_t>(tlen));
                    if (cfg.large.write_vector_bucket) {
                        // Store raw float vectors losslessly for float datasets.
                        if (!writer->AppendBlockRawF32(global_start,
                                                       cluster_id,
                                                       B_small,
                                                       a,
                                                       x_ptr,
                                                       /*ld_x_f32=*/d,
                                                       err)) {
                            return false;
                        }
                    }
                    else {
                        if (!writer->AppendBlock(global_start, cluster_id, B_small, a, /*x_u8_optional=*/nullptr,
                                                 err)) {
                            return false;
                        }
                    }
                }
                t_write += t.ElapsedSeconds();
            }

            t_read_total += t_read;
            t_rotate_total += t_rot;
            t_copy_total += t_copy;
            t_xc_total += t_xc;
            t_beam_total += t_beam;
            t_beam_root_gemm_total += t_beam_root_gemm;
            t_beam_root_update_total += t_beam_root_update;
            t_beam_expand_total += t_beam_expand;
            t_ls_total += t_ls;
            t_icm_total += t_icm;
            t_write_total += t_write;
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
            if (cfg.large.profile_timing && used_cuda_icm) {
                icm_cuda_total.upload_fixed += icm_cuda_block.upload_fixed;
                icm_cuda_total.upload_B += icm_cuda_block.upload_B;
                icm_cuda_total.icm_init += icm_cuda_block.icm_init;
                icm_cuda_total.ils_copy_perturb += icm_cuda_block.ils_copy_perturb;
                icm_cuda_total.ils_icm += icm_cuda_block.ils_icm;
                icm_cuda_total.ils_accept += icm_cuda_block.ils_accept;
                icm_cuda_total.download += icm_cuda_block.download;
            }
#endif

            const bool last_block = (start + block >= nbase);
            const std::uint64_t block_id = start / block;
            if (cfg.large.profile_timing) {
                LogInfo("  block " + std::to_string(block_id + 1) + " / " + std::to_string(blocks_total) +
                    " timing(s): read=" + FormatFloat(static_cast<float>(t_read), 4) +
                    " rotate=" + FormatFloat(static_cast<float>(t_rot), 4) +
                    " copy=" + FormatFloat(static_cast<float>(t_copy), 4) +
                    " xC=" + FormatFloat(static_cast<float>(t_xc), 4) +
                    " beam=" + FormatFloat(static_cast<float>(t_beam), 4) +
                    (use_large_root
                         ? (" (root_gemm=" + FormatFloat(static_cast<float>(t_beam_root_gemm), 4) +
                             " root_upd=" + FormatFloat(static_cast<float>(t_beam_root_update), 4) +
                             " expand=" + FormatFloat(static_cast<float>(t_beam_expand), 4) + ")")
                         : "") +
                    " ls=" + FormatFloat(static_cast<float>(t_ls), 4) +
                    " icm_ils=" + FormatFloat(static_cast<float>(t_icm), 4) +
                    " write=" + FormatFloat(static_cast<float>(t_write), 4) +
                    " total=" + FormatFloat(static_cast<float>(t_block.ElapsedSeconds()), 4));
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                if (used_cuda_icm) {
                    LogInfo("  icm_cuda timing(s): up_fixed=" + FormatFloat(
                            static_cast<float>(icm_cuda_block.upload_fixed), 4) +
                        " up_B=" + FormatFloat(static_cast<float>(icm_cuda_block.upload_B), 4) +
                        " icm_init=" + FormatFloat(static_cast<float>(icm_cuda_block.icm_init), 4) +
                        " ils_copy=" + FormatFloat(static_cast<float>(icm_cuda_block.ils_copy_perturb), 4) +
                        " ils_icm=" + FormatFloat(static_cast<float>(icm_cuda_block.ils_icm), 4) +
                        " ils_accept=" + FormatFloat(static_cast<float>(icm_cuda_block.ils_accept), 4) +
                        " down=" + FormatFloat(static_cast<float>(icm_cuda_block.download), 4));
                }
#endif
            }

            std::string msg = "Encoded block " + std::to_string(block_id + 1) + " / " + std::to_string(blocks_total);
            if (cfg.large.profile_timing) {
                msg += " (block_total_s=" + FormatFloat(static_cast<float>(t_block.ElapsedSeconds()), 4) + ")";
            }
            if (last_block) {
                LogInfoProgressDone(msg);
            }
            else {
                LogInfoProgress(msg);
            }

            const std::uint64_t next_id = start + static_cast<std::uint64_t>(nread);
            if (!CommitBlock(next_id)) {
                return false;
            }
        }

        if (async_io) {
            async_io->Stop();
        }
        if (async_writer && async_writer->Enabled()) {
            Timer t_flush;
            if (!async_writer->Finish()) {
                return false;
            }
            t_write_total += t_flush.ElapsedSeconds();
        }

        const float beam_mse = (want_mse && n_total > 0)
                                   ? static_cast<float>(mse_sum_beam / static_cast<double>(n_total))
                                   : 0.0f;
        const float final_mse = (want_mse && n_total > 0)
                                    ? static_cast<float>(mse_sum_final / static_cast<double>(n_total))
                                    : 0.0f;
        if (out_beam_mse) *out_beam_mse = beam_mse;
        if (out_final_mse) *out_final_mse = final_mse;
        if (cfg.train.log_metrics) {
            LogInfo("Base streaming(F32) beam MSE: " + FormatFloat(beam_mse, 6));
            LogInfo("Base streaming(F32) final MSE: " + FormatFloat(final_mse, 6));
        }
        if (cfg.large.profile_timing) {
            LogInfo("Base streaming(F32) total timing(s): read=" + FormatFloat(static_cast<float>(t_read_total), 3) +
                " rotate=" + FormatFloat(static_cast<float>(t_rotate_total), 3) +
                " copy=" + FormatFloat(static_cast<float>(t_copy_total), 3) +
                " xC=" + FormatFloat(static_cast<float>(t_xc_total), 3) +
                " beam=" + FormatFloat(static_cast<float>(t_beam_total), 3) +
                (use_large_root
                     ? (" (root_gemm=" + FormatFloat(static_cast<float>(t_beam_root_gemm_total), 3) +
                         " root_upd=" + FormatFloat(static_cast<float>(t_beam_root_update_total), 3) +
                         " expand=" + FormatFloat(static_cast<float>(t_beam_expand_total), 3) + ")")
                     : "") +
                " ls=" + FormatFloat(static_cast<float>(t_ls_total), 3) +
                " icm_ils=" + FormatFloat(static_cast<float>(t_icm_total), 3) +
                " write=" + FormatFloat(static_cast<float>(t_write_total), 3));
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
            LogInfo("Base streaming(F32) total icm_cuda timing(s): up_fixed=" +
                FormatFloat(static_cast<float>(icm_cuda_total.upload_fixed), 3) +
                " up_B=" + FormatFloat(static_cast<float>(icm_cuda_total.upload_B), 3) +
                " icm_init=" + FormatFloat(static_cast<float>(icm_cuda_total.icm_init), 3) +
                " ils_copy=" + FormatFloat(static_cast<float>(icm_cuda_total.ils_copy_perturb), 3) +
                " ils_icm=" + FormatFloat(static_cast<float>(icm_cuda_total.ils_icm), 3) +
                " ils_accept=" + FormatFloat(static_cast<float>(icm_cuda_total.ils_accept), 3) +
                " down=" + FormatFloat(static_cast<float>(icm_cuda_total.download), 3));
#endif
        }
        LogInfo("Base streaming(F32) wall time(s): " + FormatFloat(static_cast<float>(t_wall.ElapsedSeconds()), 3));
        return true;
    }

    // --- Public overloads delegating to the template implementation ---

    bool EncodeBaseStreamingF32(const Config& cfg,
                                const io::FvecsReader& base_reader,
                                const ColMajorMatrix<float>& R,
                                const CodebookPack& C_root,
                                const Precomp& pre_full,
                                const PrecompLargeRoot* pre_large,
                                StreamKernelProvider* kernels,
                                io::BaseBasicWriter* writer,
                                float* out_beam_mse,
                                float* out_final_mse,
                                std::string* err,
                                std::uint64_t start_id,
                                const StreamingBasicCommitHook* commit_hook) {
        return EncodeBaseStreamingF32Impl(cfg, base_reader, R, C_root, pre_full, pre_large,
                                          kernels, writer, out_beam_mse, out_final_mse, err,
                                          start_id, commit_hook);
    }

    bool EncodeBaseStreamingF32(const Config& cfg,
                                const io::FbinReader& base_reader,
                                const ColMajorMatrix<float>& R,
                                const CodebookPack& C_root,
                                const Precomp& pre_full,
                                const PrecompLargeRoot* pre_large,
                                StreamKernelProvider* kernels,
                                io::BaseBasicWriter* writer,
                                float* out_beam_mse,
                                float* out_final_mse,
                                std::string* err,
                                std::uint64_t start_id,
                                const StreamingBasicCommitHook* commit_hook) {
        return EncodeBaseStreamingF32Impl(cfg, base_reader, R, C_root, pre_full, pre_large,
                                          kernels, writer, out_beam_mse, out_final_mse, err,
                                          start_id, commit_hook);
    }
} // namespace stlq
