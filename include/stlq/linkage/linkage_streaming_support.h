#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "stlq/common/types.h"

namespace stlq {
namespace io {
class BaseListReader;
class IvfListsReader;
struct DatasetVectorReader;
}  // namespace io

namespace linkage {

struct PrefetchedClusterSpans {
    int cid = -1;
    std::uint64_t off = 0;
    std::vector<std::uint32_t> ids;
    ColMajorMatrix<std::uint8_t> x_u8;
    ColMajorMatrix<float> x_f32;
    std::vector<Code> codes;
    std::vector<float> coeffs;
};

class AsyncClusterPrefetcher {
public:
    AsyncClusterPrefetcher(const io::BaseListReader& base_list,
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
                           const char* context);

    ~AsyncClusterPrefetcher();

    AsyncClusterPrefetcher(const AsyncClusterPrefetcher&) = delete;
    AsyncClusterPrefetcher& operator=(const AsyncClusterPrefetcher&) = delete;

    void Start();
    void Stop();
    bool Pop(PrefetchedClusterSpans* out);

    double read_ivf_s() const;
    double read_raw_s() const;
    double read_base_s() const;

private:
    static long long NowNs();

    void Fail(const std::string& msg);
    void Run();

    const io::BaseListReader* base_list_ = nullptr;
    const io::IvfListsReader* ivf_lists_ = nullptr;
    int nlist_ = 0;
    const std::vector<std::uint8_t>* skip_done_ = nullptr;
    bool has_raw_f32_ = false;
    bool has_raw_u8_ = false;
    int depth_ = 2;
    bool profile_timing_ = false;

    std::atomic<bool>* ok_ = nullptr;
    std::string* first_err_ = nullptr;
    std::mutex* err_mu_ = nullptr;
    const char* context_ = nullptr;

    std::atomic<bool> stop_{false};
    std::atomic<bool> done_{false};
    std::thread worker_;

    mutable std::mutex mu_;
    std::condition_variable cv_not_empty_;
    std::condition_variable cv_not_full_;
    std::deque<PrefetchedClusterSpans> queue_;

    long long read_ivf_ns_ = 0;
    long long read_raw_ns_ = 0;
    long long read_base_ns_ = 0;
};

struct ClusterCkptStatsV1 {
    std::uint64_t real = 0;
    std::uint64_t linkaged = 0;
    double depth_sum = 0.0;
    std::int32_t max_depth = 0;
    std::int32_t reserved0 = 0;
    double mse_sum = 0.0;
    double mse_min = 0.0;
    double mse_max = 0.0;
};

struct ClusterCheckpoint {
    bool enabled = false;
    int nlist = 0;
    std::filesystem::path dir;
    std::filesystem::path done_path;
    std::filesystem::path stats_path;
    std::vector<std::uint8_t> done;

#if defined(_WIN32)
    std::fstream done_out;
    std::fstream stats_out;
    std::mutex mu;
#else
    int fd_done = -1;
    int fd_stats = -1;
#endif

    static bool ReadFileBytes(const std::filesystem::path& p, std::vector<std::uint8_t>* out);

    bool Open(const std::string& dir_str, int n, std::string* err);
    [[nodiscard]] bool IsDone(int cid) const;

    bool LoadStatsInto(std::vector<std::uint64_t>* cluster_real,
                       std::vector<std::uint64_t>* cluster_linkaged,
                       std::vector<double>* cluster_depth_sum,
                       std::vector<int>* cluster_max_depth,
                       std::vector<double>* cluster_mse_sum,
                       std::vector<double>* cluster_mse_min,
                       std::vector<double>* cluster_mse_max,
                       std::string* err) const;

    bool MarkDone(int cid, const ClusterCkptStatsV1& s, std::string* err);
};

struct ClusterWorkBuf {
    std::vector<std::uint32_t> ids;
    ColMajorMatrix<std::uint8_t> x_u8;
    ColMajorMatrix<float> x_f32;
    ColMajorMatrix<float> Xrot;

    ColMajorMatrix<Code> B_small;
    ColMajorMatrix<float> a;
    ColMajorMatrix<FullCode> B_full;
    std::vector<RootCode> code0_root;

    std::vector<int> cols_local;
    std::vector<std::uint32_t> knn_ids_flat;
    std::vector<float> knn_dists_flat;
    ColMajorMatrix<float> X_virt;
    ColMajorMatrix<FullCode> B_virt;
    ColMajorMatrix<float> a_virt;
    ColMajorMatrix<float> R_full_shared;
    ColMajorMatrix<float> R_virt;
    LinkageStructure linkage_bad;

    LinkageStructure::Cluster cluster_out;
    std::vector<std::uint32_t> real_ids_depth_order;
    std::vector<std::uint32_t> parent_u32;
    std::vector<std::uint32_t> depth_offsets_u32;
    std::vector<std::uint8_t> codes_small_depth;
    std::vector<float> coeffs_small_depth;
    std::vector<std::uint8_t> code0_one_depth;
    std::vector<float> a0_depth;
    std::vector<std::uint8_t> virt_codes_small;
    std::vector<float> virt_coeffs_small;
    std::vector<float> virt_a0;

    std::vector<float> a_layer_major;
    std::vector<std::uint8_t> is_linkage;
    std::vector<std::int8_t> q_tmp;

    std::vector<std::vector<std::uint8_t>> lens_root;
    std::vector<std::vector<std::uint8_t>> lens_linkage;
    std::vector<std::vector<std::uint8_t>> payload_root;
    std::vector<std::vector<std::uint8_t>> payload_linkage;

    std::vector<std::uint32_t> B_code1_layer_major;
    std::vector<float> g0s_root_small;

    void EnsureReal(int d, int n_real, int m, int m_codes) {
        x_u8.rows = d;
        x_u8.cols = n_real;
        if (x_u8.data.size() < static_cast<std::size_t>(d) * static_cast<std::size_t>(n_real)) {
            x_u8.data.resize(static_cast<std::size_t>(d) * static_cast<std::size_t>(n_real));
        }

        x_f32.rows = d;
        x_f32.cols = n_real;
        if (x_f32.data.size() < static_cast<std::size_t>(d) * static_cast<std::size_t>(n_real)) {
            x_f32.data.resize(static_cast<std::size_t>(d) * static_cast<std::size_t>(n_real));
        }

        Xrot.rows = d;
        Xrot.cols = n_real;
        if (Xrot.data.size() < static_cast<std::size_t>(d) * static_cast<std::size_t>(n_real)) {
            Xrot.data.resize(static_cast<std::size_t>(d) * static_cast<std::size_t>(n_real));
        }

        B_small.rows = m_codes;
        B_small.cols = n_real;
        if (B_small.data.size() < static_cast<std::size_t>(m_codes) * static_cast<std::size_t>(n_real)) {
            B_small.data.resize(static_cast<std::size_t>(m_codes) * static_cast<std::size_t>(n_real));
        }

        a.rows = m;
        a.cols = n_real;
        if (a.data.size() < static_cast<std::size_t>(m) * static_cast<std::size_t>(n_real)) {
            a.data.resize(static_cast<std::size_t>(m) * static_cast<std::size_t>(n_real));
        }

        B_full.rows = m;
        B_full.cols = n_real;
        if (B_full.data.size() < static_cast<std::size_t>(m) * static_cast<std::size_t>(n_real)) {
            B_full.data.resize(static_cast<std::size_t>(m) * static_cast<std::size_t>(n_real));
        }
    }

    void EnsureVirt(int d, int n_virt, int m, int m_codes) {
        X_virt.rows = d;
        X_virt.cols = n_virt;
        if (X_virt.data.size() < static_cast<std::size_t>(d) * static_cast<std::size_t>(n_virt)) {
            X_virt.data.resize(static_cast<std::size_t>(d) * static_cast<std::size_t>(n_virt));
        }
        B_virt.rows = m;
        B_virt.cols = n_virt;
        if (B_virt.data.size() < static_cast<std::size_t>(m) * static_cast<std::size_t>(n_virt)) {
            B_virt.data.resize(static_cast<std::size_t>(m) * static_cast<std::size_t>(n_virt));
        }
        a_virt.rows = m;
        a_virt.cols = n_virt;
        if (a_virt.data.size() < static_cast<std::size_t>(m) * static_cast<std::size_t>(n_virt)) {
            a_virt.data.resize(static_cast<std::size_t>(m) * static_cast<std::size_t>(n_virt));
        }
        (void)m_codes;
    }

    void ClearPerCluster() {
        ids.clear();
        cols_local.clear();
        knn_ids_flat.clear();
        knn_dists_flat.clear();

        Xrot.cols = 0;
        B_small.cols = 0;
        a.cols = 0;
        B_full.cols = 0;
        code0_root.clear();
        X_virt.cols = 0;
        B_virt.cols = 0;
        a_virt.cols = 0;
        linkage_bad.clusters.clear();
        cluster_out.n_real = 0;
        cluster_out.n_virtual = 0;

        a_layer_major.clear();
        is_linkage.clear();
        q_tmp.clear();
        B_code1_layer_major.clear();
        g0s_root_small.clear();
    }
};

bool ReaderCanReadU8(const io::DatasetVectorReader* reader);
bool ReaderCanReadF32(const io::DatasetVectorReader* reader);

Precomp BuildPrecompMetaOnly(const CodebookPack& pack);

bool BuildPrecompInitLinkageFixedRootTinyCpu(const CodebookPack& C_root,
                                             int forced_root_code,
                                             Precomp* out,
                                             std::string* err);

bool SolveSymPosdefCholeskyRetry(int m, const double* A_in, const double* b_in, double* x_out);

}  // namespace linkage
}  // namespace stlq
