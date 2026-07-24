#include "stlq/linkage/linkage_streaming_support.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#endif

#include "stlq/core/blas.h"
#include "stlq/core/lapack.h"
#include "stlq/core/threading.h"
#include "stlq/io/base_list_store.h"
#include "stlq/io/dataset_reader_factory.h"
#include "stlq/io/ivf_lists.h"

namespace stlq::linkage {

AsyncClusterPrefetcher::AsyncClusterPrefetcher(const io::BaseListReader& base_list,
                                               const io::IvfListsReader& ivf_lists,
                                               int nlist,
                                               const std::vector<std::uint8_t>* skip_done,
                                               bool has_raw_f32,
                                               bool has_raw_u8,
                                               int depth,
                                               bool profile_timing,
                                               std::atomic<bool>* ok,
                                               std::string* first_err,
                                               std::mutex* err_mu,
                                               const char* context)
    : base_list_(&base_list),
      ivf_lists_(&ivf_lists),
      nlist_(nlist),
      skip_done_(skip_done),
      has_raw_f32_(has_raw_f32),
      has_raw_u8_(has_raw_u8),
      depth_(std::max(1, depth)),
      profile_timing_(profile_timing),
      ok_(ok),
      first_err_(first_err),
      err_mu_(err_mu),
      context_(context ? context : "AsyncClusterPrefetcher") {
}

AsyncClusterPrefetcher::~AsyncClusterPrefetcher() {
    Stop();
}

void AsyncClusterPrefetcher::Start() {
    stop_.store(false);
    done_.store(false);
    worker_ = std::thread([this]() { this->Run(); });
}

void AsyncClusterPrefetcher::Stop() {
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

bool AsyncClusterPrefetcher::Pop(PrefetchedClusterSpans* out) {
    if (!out) return false;
    std::unique_lock<std::mutex> lock(mu_);
    cv_not_empty_.wait(lock, [&]() {
        return stop_.load() || !queue_.empty() || done_.load() || (ok_ && !ok_->load());
    });
    if (stop_.load() || (ok_ && !ok_->load())) {
        return false;
    }
    if (queue_.empty()) {
        return false;
    }
    *out = std::move(queue_.front());
    queue_.pop_front();
    cv_not_full_.notify_one();
    return true;
}

double AsyncClusterPrefetcher::read_ivf_s() const {
    return static_cast<double>(read_ivf_ns_) * 1e-9;
}

double AsyncClusterPrefetcher::read_raw_s() const {
    return static_cast<double>(read_raw_ns_) * 1e-9;
}

double AsyncClusterPrefetcher::read_base_s() const {
    return static_cast<double>(read_base_ns_) * 1e-9;
}

long long AsyncClusterPrefetcher::NowNs() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now().time_since_epoch()).count();
}

void AsyncClusterPrefetcher::Fail(const std::string& msg) {
    if (ok_) ok_->store(false);
    if (!first_err_ || !err_mu_) return;
    std::lock_guard<std::mutex> guard(*err_mu_);
    if (first_err_->empty()) {
        if (!msg.empty()) {
            *first_err_ = msg;
        } else {
            *first_err_ = std::string(context_) + ": failed.";
        }
    }
}

void AsyncClusterPrefetcher::Run() {
    std::string local_err;
    io::BaseListThreadReader base_thr;
    io::IvfListsThreadReader ivf_thr;
    if (!base_thr.OpenFrom(*base_list_, &local_err)) {
        Fail(local_err);
        done_.store(true);
        cv_not_empty_.notify_all();
        return;
    }
    if (!ivf_thr.OpenFrom(*ivf_lists_, &local_err)) {
        Fail(local_err);
        done_.store(true);
        cv_not_empty_.notify_all();
        return;
    }

    for (int cid = 0; cid < nlist_; ++cid) {
        if (stop_.load() || (ok_ && !ok_->load())) break;

        if (skip_done_ && cid >= 0 && cid < static_cast<int>(skip_done_->size()) &&
            (*skip_done_)[static_cast<std::size_t>(cid)] != 0) {
            continue;
        }
        local_err.clear();

        PrefetchedClusterSpans item;
        item.cid = cid;
        const long long t0_ivf = profile_timing_ ? NowNs() : 0;
        if (!ivf_thr.ReadList(*ivf_lists_, cid, &item.ids, &local_err)) {
            Fail(local_err);
            break;
        }
        item.off = ivf_lists_->Offset(cid);
        if (profile_timing_) {
            read_ivf_ns_ += (NowNs() - t0_ivf);
        }

        const int n_real = static_cast<int>(item.ids.size());
        if (n_real > 0) {
            const long long t0_raw = profile_timing_ ? NowNs() : 0;
            if (has_raw_f32_) {
                if (!base_thr.ReadRawF32Span(item.off, static_cast<std::uint32_t>(n_real), &item.x_f32,
                                             &local_err)) {
                    Fail(local_err);
                    break;
                }
            } else if (has_raw_u8_) {
                if (!base_thr.ReadRawU8Span(item.off, static_cast<std::uint32_t>(n_real), &item.x_u8,
                                            &local_err)) {
                    Fail(local_err);
                    break;
                }
            } else {
                Fail(std::string(context_) + ": async IO requires list-order raw.");
                break;
            }
            if (profile_timing_) {
                read_raw_ns_ += (NowNs() - t0_raw);
            }

            const long long t0_base = profile_timing_ ? NowNs() : 0;
            if (!base_thr.ReadCodesSpan(item.off, static_cast<std::uint32_t>(n_real), &item.codes, &local_err)) {
                Fail(local_err);
                break;
            }
            if (!base_thr.ReadCoeffsSpan(item.off, static_cast<std::uint32_t>(n_real), &item.coeffs, &local_err)) {
                Fail(local_err);
                break;
            }
            if (profile_timing_) {
                read_base_ns_ += (NowNs() - t0_base);
            }
        }

        std::unique_lock<std::mutex> lock(mu_);
        cv_not_full_.wait(lock, [&]() {
            return stop_.load() || done_.load() || (ok_ && !ok_->load()) ||
                   static_cast<int>(queue_.size()) < depth_;
        });
        if (stop_.load() || (ok_ && !ok_->load())) break;
        queue_.push_back(std::move(item));
        cv_not_empty_.notify_one();
    }

    done_.store(true);
    cv_not_empty_.notify_all();
}

bool ClusterCheckpoint::ReadFileBytes(const std::filesystem::path& p, std::vector<std::uint8_t>* out) {
    if (!out) return false;
    std::ifstream in(p, std::ios::binary);
    if (!in.is_open()) return false;
    in.seekg(0, std::ios::end);
    const std::streamoff sz = in.tellg();
    if (sz < 0) return false;
    in.seekg(0, std::ios::beg);
    out->assign(static_cast<std::size_t>(sz), 0);
    if (sz > 0) {
        in.read(reinterpret_cast<char*>(out->data()), sz);
        if (!in) return false;
    }
    return true;
}

bool ClusterCheckpoint::Open(const std::string& dir_str, int n, std::string* err) {
    enabled = true;
    nlist = n;
    dir = std::filesystem::path(dir_str);
    done_path = dir / "ckpt_linkage_done.u8";
    stats_path = dir / "ckpt_linkage_stats_v1.bin";

    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        if (err) *err = "Linkage checkpoint: failed to create dir: " + dir.string();
        return false;
    }

    std::vector<std::uint8_t> bytes;
    if (std::filesystem::exists(done_path, ec) && ReadFileBytes(done_path, &bytes) &&
        static_cast<int>(bytes.size()) == nlist) {
        done = std::move(bytes);
    } else {
        done.assign(static_cast<std::size_t>(nlist), 0);
        std::ofstream out(done_path, std::ios::binary | std::ios::trunc);
        if (!out.is_open()) {
            if (err) *err = "Linkage checkpoint: failed to create done file: " + done_path.string();
            return false;
        }
        if (nlist > 0) {
            out.write(reinterpret_cast<const char*>(done.data()), static_cast<std::streamsize>(done.size()));
            if (!out) {
                if (err) *err = "Linkage checkpoint: failed to write done file.";
                return false;
            }
        }
    }

    const std::uint64_t want_stats_bytes =
        static_cast<std::uint64_t>(nlist) * static_cast<std::uint64_t>(sizeof(ClusterCkptStatsV1));
    if (!std::filesystem::exists(stats_path, ec)) {
        std::ofstream out(stats_path, std::ios::binary | std::ios::trunc);
        if (!out.is_open()) {
            if (err) *err = "Linkage checkpoint: failed to create stats file: " + stats_path.string();
            return false;
        }
    }
    try {
        std::filesystem::resize_file(stats_path, static_cast<std::uintmax_t>(want_stats_bytes));
    } catch (...) {
        if (err) *err = "Linkage checkpoint: failed to resize stats file: " + stats_path.string();
        return false;
    }

#if defined(_WIN32)
    done_out.open(done_path.string(), std::ios::binary | std::ios::in | std::ios::out);
    stats_out.open(stats_path.string(), std::ios::binary | std::ios::in | std::ios::out);
    if (!done_out.is_open() || !stats_out.is_open()) {
        if (err) *err = "Linkage checkpoint: failed to open checkpoint streams.";
        return false;
    }
#else
    fd_done = ::open(done_path.c_str(), O_RDWR | O_CREAT, 0644);
    fd_stats = ::open(stats_path.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd_done < 0 || fd_stats < 0) {
        if (err) *err = "Linkage checkpoint: failed to open checkpoint files.";
        return false;
    }
#endif
    return true;
}

bool ClusterCheckpoint::IsDone(int cid) const {
    if (!enabled) return false;
    if (cid < 0 || cid >= nlist) return false;
    return done[static_cast<std::size_t>(cid)] != 0;
}

bool ClusterCheckpoint::LoadStatsInto(std::vector<std::uint64_t>* cluster_real,
                                      std::vector<std::uint64_t>* cluster_linkaged,
                                      std::vector<double>* cluster_depth_sum,
                                      std::vector<int>* cluster_max_depth,
                                      std::vector<double>* cluster_mse_sum,
                                      std::vector<double>* cluster_mse_min,
                                      std::vector<double>* cluster_mse_max,
                                      std::string* err) const {
    if (!enabled) return true;
    if (!cluster_real || !cluster_linkaged || !cluster_depth_sum || !cluster_max_depth ||
        !cluster_mse_sum || !cluster_mse_min || !cluster_mse_max) {
        if (err) *err = "Linkage checkpoint: LoadStatsInto invalid outputs.";
        return false;
    }
    std::ifstream in(stats_path, std::ios::binary);
    if (!in.is_open()) {
        if (err) *err = "Linkage checkpoint: failed to open stats for read: " + stats_path.string();
        return false;
    }
    for (int cid = 0; cid < nlist; ++cid) {
        ClusterCkptStatsV1 s;
        in.read(reinterpret_cast<char*>(&s), sizeof(s));
        if (!in) {
            if (err) *err = "Linkage checkpoint: stats read failed.";
            return false;
        }
        if (done[static_cast<std::size_t>(cid)] == 0) {
            continue;
        }
        (*cluster_real)[static_cast<std::size_t>(cid)] = s.real;
        (*cluster_linkaged)[static_cast<std::size_t>(cid)] = s.linkaged;
        (*cluster_depth_sum)[static_cast<std::size_t>(cid)] = s.depth_sum;
        (*cluster_max_depth)[static_cast<std::size_t>(cid)] = static_cast<int>(s.max_depth);
        (*cluster_mse_sum)[static_cast<std::size_t>(cid)] = s.mse_sum;
        (*cluster_mse_min)[static_cast<std::size_t>(cid)] = s.mse_min;
        (*cluster_mse_max)[static_cast<std::size_t>(cid)] = s.mse_max;
    }
    return true;
}

bool ClusterCheckpoint::MarkDone(int cid, const ClusterCkptStatsV1& s, std::string* err) {
    if (!enabled) return true;
    if (cid < 0 || cid >= nlist) return false;

#if defined(_WIN32)
    std::lock_guard<std::mutex> lock(mu);
    const std::streamoff off_stats = static_cast<std::streamoff>(
        static_cast<std::uint64_t>(cid) * static_cast<std::uint64_t>(sizeof(ClusterCkptStatsV1)));
    stats_out.seekp(off_stats, std::ios::beg);
    stats_out.write(reinterpret_cast<const char*>(&s), static_cast<std::streamsize>(sizeof(s)));
    if (!stats_out) {
        if (err) *err = "Linkage checkpoint: stats write failed.";
        return false;
    }
    const std::uint8_t one = 1;
    const std::streamoff off_done = static_cast<std::streamoff>(cid);
    done_out.seekp(off_done, std::ios::beg);
    done_out.write(reinterpret_cast<const char*>(&one), 1);
    if (!done_out) {
        if (err) *err = "Linkage checkpoint: done write failed.";
        return false;
    }
#else
    const std::uint64_t off_stats =
        static_cast<std::uint64_t>(cid) * static_cast<std::uint64_t>(sizeof(ClusterCkptStatsV1));
    const auto* ps = reinterpret_cast<const std::uint8_t*>(&s);
    std::size_t remaining = sizeof(ClusterCkptStatsV1);
    std::uint64_t cur = off_stats;
    while (remaining > 0) {
        const ssize_t w = ::pwrite(fd_stats, ps, remaining, static_cast<off_t>(cur));
        if (w <= 0) {
            if (err) *err = "Linkage checkpoint: pwrite(stats) failed.";
            return false;
        }
        remaining -= static_cast<std::size_t>(w);
        ps += w;
        cur += static_cast<std::uint64_t>(w);
    }
    const std::uint8_t one = 1;
    if (::pwrite(fd_done, &one, 1, static_cast<off_t>(cid)) != 1) {
        if (err) *err = "Linkage checkpoint: pwrite(done) failed.";
        return false;
    }
#endif
    done[static_cast<std::size_t>(cid)] = 1;
    return true;
}

bool ReaderCanReadU8(const io::DatasetVectorReader* reader) {
    return reader && reader->format == io::VectorFileFormat::kBvecs;
}

bool ReaderCanReadF32(const io::DatasetVectorReader* reader) {
    return reader && (reader->format == io::VectorFileFormat::kFvecs ||
                      reader->format == io::VectorFileFormat::kFbin);
}

Precomp BuildPrecompMetaOnly(const CodebookPack& pack) {
    Precomp pre;
    pre.d = pack.d;
    pre.m = static_cast<int>(pack.books.size());
    pre.h_vec = pack.h_vec;
    pre.offsets.assign(static_cast<std::size_t>(std::max(0, pre.m)), 0);
    int H = 0;
    for (int l = 0; l < pre.m; ++l) {
        pre.offsets[static_cast<std::size_t>(l)] = H;
        const int hl = (l < static_cast<int>(pre.h_vec.size())) ? pre.h_vec[static_cast<std::size_t>(l)] : 0;
        H += std::max(0, hl);
    }
    pre.H = H;
    pre.build_tag = pack.build_tag;
    return pre;
}

bool BuildPrecompInitLinkageFixedRootTinyCpu(const CodebookPack& C_root,
                                             int forced_root_code,
                                             Precomp* out,
                                             std::string* err) {
    if (!out || C_root.books.empty()) {
        if (err) *err = "BuildPrecompInitLinkageFixedRootTinyCpu: invalid args.";
        return false;
    }
    const int m = static_cast<int>(C_root.books.size());
    const int d = C_root.books.front().rows;
    if (m <= 0 || d <= 0 || C_root.books.front().cols <= 0) {
        if (err) *err = "BuildPrecompInitLinkageFixedRootTinyCpu: invalid d/m.";
        return false;
    }
    const int h0 = C_root.books.front().cols;
    if (forced_root_code < 0 || forced_root_code >= h0) {
        if (err) *err = "BuildPrecompInitLinkageFixedRootTinyCpu: forced_root_code out of range.";
        return false;
    }

    std::vector<int> h_vec(static_cast<std::size_t>(m), 0);
    h_vec[0] = 1;
    int H = 1;
    for (int l = 1; l < m; ++l) {
        const int hl = C_root.books[static_cast<std::size_t>(l)].cols;
        h_vec[static_cast<std::size_t>(l)] = hl;
        H += hl;
    }

    out->C_all = ColMajorMatrix<float>(d, H);
    out->G = ColMajorMatrix<float>(H, H);
    out->offsets.assign(static_cast<std::size_t>(m), 0);
    out->flat_layer.assign(static_cast<std::size_t>(H), 0);
    out->invnorm_flat.assign(static_cast<std::size_t>(H), 0.0f);
    out->h_vec = h_vec;
    out->H = H;
    out->d = d;
    out->m = m;
    out->build_tag = C_root.build_tag;

    int col_offset = 0;
    for (int l = 0; l < m; ++l) {
        out->offsets[static_cast<std::size_t>(l)] = col_offset;
        const int hl = h_vec[static_cast<std::size_t>(l)];
        if (hl <= 0) continue;

        if (l == 0) {
            const float* src = C_root.books.front().Col(forced_root_code);
            float* dst = out->C_all.Col(0);
            float norm_sq = 0.0f;
            for (int r = 0; r < d; ++r) {
                dst[r] = src[r];
                norm_sq += src[r] * src[r];
            }
            const float inv_norm = norm_sq > 0.0f ? 1.0f / std::sqrt(norm_sq) : 0.0f;
            out->flat_layer[0] = 0;
            out->invnorm_flat[0] = inv_norm;
        } else {
            const auto& book = C_root.books[static_cast<std::size_t>(l)];
            for (int c = 0; c < book.cols; ++c) {
                const float* src = book.Col(c);
                float* dst = out->C_all.Col(col_offset + c);
                float norm_sq = 0.0f;
                for (int r = 0; r < d; ++r) {
                    dst[r] = src[r];
                    norm_sq += src[r] * src[r];
                }
                const float inv_norm = norm_sq > 0.0f ? 1.0f / std::sqrt(norm_sq) : 0.0f;
                out->flat_layer[col_offset + c] = l;
                out->invnorm_flat[col_offset + c] = inv_norm;
            }
        }
        col_offset += hl;
    }

    {
        ScopedBlasThreads blas_scope(1);
        Gemm(true, false, 1.0f, out->C_all, out->C_all, 0.0f, &out->G);
    }
    return true;
}

bool SolveSymPosdefCholeskyRetry(int m, const double* A_in, const double* b_in, double* x_out) {
    if (m <= 0 || !A_in || !b_in || !x_out) {
        return false;
    }
    thread_local std::vector<double> A_cm;
    thread_local std::vector<double> b;
    A_cm.resize(static_cast<std::size_t>(m) * static_cast<std::size_t>(m));
    b.resize(static_cast<std::size_t>(m));

    auto try_once = [&](double reg) -> bool {
        for (int r = 0; r < m; ++r) {
            for (int c = 0; c < m; ++c) {
                A_cm[static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * static_cast<std::size_t>(m)] =
                    A_in[static_cast<std::size_t>(r) * static_cast<std::size_t>(m) + static_cast<std::size_t>(c)];
            }
        }
        for (int i = 0; i < m; ++i) {
            A_cm[static_cast<std::size_t>(i) + static_cast<std::size_t>(i) * static_cast<std::size_t>(m)] += reg;
        }
        std::copy(b_in, b_in + m, b.begin());

        if (lapack::DpotrfU(m, A_cm.data(), m) != 0) {
            return false;
        }
        if (lapack::DpotrsU(m, /*nrhs=*/1, A_cm.data(), m, b.data(), m) != 0) {
            return false;
        }
        std::copy(b.begin(), b.end(), x_out);
        return true;
    };

    if (try_once(0.0)) return true;
    double reg = 1e-6;
    for (int t = 0; t < 3; ++t) {
        if (try_once(reg)) return true;
        reg *= 10.0;
    }
    return false;
}

}  // namespace stlq::linkage
